import torch
import pickle
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import warnings

def main():
    global_path = './'
    vocab_path = global_path + 'vocab.pkl'
    model_path = global_path + 'model_reduced_dataset.pkl'
    images_count = 6
    images_paths = [global_path + str(i + 1) + '.jpg' for i in range(images_count)]
    
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    model = Autoencoder(len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path))
    nice_check(vocab, model, images_paths, device)

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    def __len__(self):
        return len(self.word2idx)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained = False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embed_size = 256, hidden_size = 512, num_layers = 1, max_seq_length = 20):
        super(type(self), self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)
    def forward(self, images, captions, lengths):
        return self.decoder(self.encoder(images), captions, lengths)

def load_image(image_path, transform = True):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    if transform:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                             (0.229, 0.224, 0.225))])
        image = transform(image).unsqueeze(0)
    return image

def describe_images(vocab, model, images, device):
    num_images = len(images)
    model.eval()
    texts = []
    for i in range(num_images):
        image = images[i]
        image_tensor = image.to(device)
        feature = model.encoder(image_tensor)
        sampled_ids = model.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        description = ' '.join(sampled_caption)
        texts.append(description)
    return texts

def nice_check(vocab, model, images_paths, device):
    my_images = [load_image(i) for i in images_paths]
    texts = describe_images(vocab, model, my_images, device)
    for i in range(len(texts)):
        print(images_paths[i] + ": " + texts[i])

if __name__ == "__main__":
    main()

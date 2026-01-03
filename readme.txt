Наумкин Владимир С01-119.

sem1 - решения двух ДЗ осеннего семестра (+ методичка https://github.com/andriygav/MLbook/pull/3 для автомата)

sem2 - решения 8/9 задач весеннего семестра.
Хронология очных сдач:
14.03.25 - 1 и 2 задачи; замечания (для следующих задач) - надо было обучить лучшую модель, а также во 2й задаче для исследования зависимости качества от размер словаря надо было не увеличивать тренировочный датасет, а убирать из словаря редкие слова (см. min_count в 3 задаче или threshold в 5)
24.03.25 - 3 и 4 задачи; замечание к 3 задаче - batchnorm расположил не в том месте, надо было после каждого слоя, кроме выходного
04.04.25 - 5 и 6 (без замечаний)
11.04.25 - 7 и 8 (без замечаний)


Репозиторий из приватного стал публичным 11.04.25 после сдачи 7 и 8 задач.

P.S: sem2/task1/task4 - в def log_likelihood не учёл sigma (только mu), надо так (https://github.com/nerett/machine-learning/blob/main/applied_models/4-VAE/task4.ipynb):

    @staticmethod
    def log_likelihood(x_true, x_distr):
        """
        Compute the log-likelihood of x_true under a normal distribution defined by x_distr.
        Args:
            x_true: Tensor - shape (batch_size, input_dim), ground truth samples.
            x_distr: Tensor - shape (batch_size, num_samples, 2 * input_dim),
                    where first half represents mean, second half represents std (sigma).
        Returns:
            Tensor - shape (batch_size, num_samples), log-likelihood values.
        """
        batch_size, num_samples, total_dim = x_distr.shape
        real_input_dim = total_dim // 2

        mu = x_distr[:, :, :real_input_dim]
        sigma = torch.nn.Softplus()(x_distr[:, :, real_input_dim:])

        if x_true.shape[1] != real_input_dim:
            raise ValueError(f"Mismatch: x_true.shape[1] = {x_true.shape[1]}, expected {real_input_dim}")

        x_true = x_true.unsqueeze(1).expand(-1, num_samples, -1)  # Shape: (batch_size, num_samples, real_input_dim)

        log_likelihood = -0.5 * torch.sum(((x_true - mu) ** 2) / (sigma ** 2) +
                                        2 * torch.log(sigma) +
                                        math.log(2 * math.pi), dim=2)

        return log_likelihood

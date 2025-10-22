import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class VoiceBankDataset(Dataset):
    """
    Класс для работы с датасетом VoiceBank.
    Обеспечивает загрузку и предобработку аудиофайлов.
    """
    def __init__(self, noisy_dir, clean_dir, n_fft=512, hop_length=128):
        """
        Инициализация датасета.

        Параметры:
            noisy_dir (str): Путь к директории с зашумленными аудиофайлами
            clean_dir (str): Путь к директории с чистыми аудиофайлами
            n_fft (int): Размер окна для FFT (по умолчанию 512)
            hop_length (int): Шаг между окнами для STFT (по умолчанию 128)
        """
        super().__init__()
        self.noisy_paths = sorted([
            os.path.join(noisy_dir, f)
            for f in os.listdir(noisy_dir)
            if f.endswith('.wav')
        ])
        self.clean_dir = clean_dir
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=2
        )

    def __len__(self):
        """
        Возвращает количество образцов в датасете.

        Возвращает:
            int: Количество аудиофайлов в датасете
        """
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        """
        Загружает и обрабатывает аудиофайл по индексу.

        Параметры:
            idx (int): Индекс образца в датасете

        Возвращает:
            tuple: Кортеж из:
                - Матрица спектрограммы зашумленного сигнала
                - Матрица спектрограммы чистого сигнала
                - Зашумленный аудиосигнал (временной ряд)
                - Чистый аудиосигнал (временной ряд)
        """
        noisy_path = self.noisy_paths[idx]
        clean_path = os.path.join(self.clean_dir, os.path.basename(noisy_path))

        # Загрузка аудиофайлов
        noisy_wav, sr = torchaudio.load(noisy_path)
        clean_wav, _ = torchaudio.load(clean_path)

        # Конвертация в моно, если нужно
        if noisy_wav.size(0) > 1:
            noisy_wav = noisy_wav.mean(dim=0, keepdim=True)
            clean_wav = clean_wav.mean(dim=0, keepdim=True)
        noisy_wav = noisy_wav.squeeze(0)
        clean_wav = clean_wav.squeeze(0)

        # Вычисление спектрограмм
        noisy_spec = self.stft(noisy_wav.unsqueeze(0))
        clean_spec = self.stft(clean_wav.unsqueeze(0))
        noisy_mag = torch.sqrt(noisy_spec + 1e-8).squeeze(0)
        clean_mag = torch.sqrt(clean_spec + 1e-8).squeeze(0)

        return noisy_mag, clean_mag, noisy_wav, clean_wav


def pad_and_crop_collate(batch):
    """
    Функция для выравнивания размеров спектрограмм и аудиосигналов в батче.

    Параметры:
        batch (list): Список образцов из датасета

    Возвращает:
        tuple: Кортеж из:
            - Выровненные спектрограммы зашумленных сигналов
            - Выровненные спектрограммы чистых сигналов
            - Выровненные зашумленные аудиосигналы
            - Выровненные чистые аудиосигналы
    """
    # Распаковка батча
    noisy_mag_list, clean_mag_list, noisy_wav_list, clean_wav_list = zip(*batch)

    # Выравнивание размеров спектрограмм
    orig_F = noisy_mag_list[0].shape[0]
    F8 = (orig_F // 8) * 8
    min_T = min(m.shape[1] for m in noisy_mag_list)
    T8 = (min_T // 8) * 8

    # Выравнивание длины аудиосигналов
    max_wav_len = max(w.shape[0] for w in noisy_wav_list)

    nm_batch, cm_batch, nw_batch, cw_batch = [], [], [], []
    for nm, cm, nw, cw in zip(noisy_mag_list, clean_mag_list, noisy_wav_list, clean_wav_list):
        # Обрезка спектрограмм
        nm_c = nm[:F8, :T8]
        cm_c = cm[:F8, :T8]

        # Дополнение аудиосигналов
        pad_amt = max_wav_len - nw.shape[0]
        nw_p = F.pad(nw, (0, pad_amt))
        cw_p = F.pad(cw, (0, pad_amt))

        nm_batch.append(nm_c)
        cm_batch.append(cm_c)
        nw_batch.append(nw_p)
        cw_batch.append(cw_p)

    # Сборка батча
    batch_noisy_mag = torch.stack(nm_batch)
    batch_clean_mag = torch.stack(cm_batch)
    batch_noisy_wav = torch.stack(nw_batch)
    batch_clean_wav = torch.stack(cw_batch)

    return batch_noisy_mag, batch_clean_mag, batch_noisy_wav, batch_clean_wav


class UNetSpectrogramDenoiser(nn.Module):
    """
    U-Net модель для очистки аудио по спектрограммам.
    """
    def __init__(self, in_channels=1, base_channels=16):
        """
        Инициализация модели U-Net.

        Параметры:
            in_channels (int): Количество входных каналов (по умолчанию 1)
            base_channels (int): Базовое количество каналов (по умолчанию 16)
        """
        super().__init__()
        
        # Энкодер
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Боттлнек
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
        )

        # Декодер
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Финальный слой
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
            nn.Softplus()  # Гарантирует неотрицательный выход
        )

    def crop(self, enc_tensor, dec_tensor_size):
        """
        Обрезает тензор энкодера для соответствия размеру тензора декодера.

        Параметры:
            enc_tensor (torch.Tensor): Тензор из энкодера
            dec_tensor_size (tuple): Размер тензора декодера

        Возвращает:
            crop (torch.Tensor): Обрезанный тензор
        """
        _, _, H, W = enc_tensor.size()
        diff_H = H - dec_tensor_size[0]
        diff_W = W - dec_tensor_size[1]
        
        start_H = diff_H // 2
        start_W = diff_W // 2
        end_H = H - (diff_H - start_H)
        end_W = W - (diff_W - start_W)
        
        return enc_tensor[:, :, start_H:end_H, start_W:end_W]

    def forward(self, x):
        """
        Прямой проход модели.

        Параметры:
            x (torch.Tensor): Входной тензор (спектрограмма)

        Возвращает:
            out (torch.Tensor): Очищенная спектрограмма
        """
        # Подготовка входа
        x = x.unsqueeze(1)

        # Энкодер
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Боттлнек
        b = self.bottleneck(p3)

        # Декодер с skip-соединениями
        d3 = self.up3(b)
        e3_cropped = self.crop(e3, d3.size()[2:])
        d3 = torch.cat([d3, e3_cropped], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_cropped = self.crop(e2, d2.size()[2:])
        d2 = torch.cat([d2, e2_cropped], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_cropped = self.crop(e1, d1.size()[2:])
        d1 = torch.cat([d1, e1_cropped], dim=1)
        d1 = self.dec1(d1)

        # Финальный слой
        out = self.final_conv(d1)
        return out.squeeze(1)


def compute_snr(clean, enhanced, eps=1e-8):
    """
    Вычисляет SNR (Signal-to-Noise Ratio) между сигналами.

    Параметры:
        clean (np.ndarray): Исходный чистый сигнал
        enhanced (np.ndarray): Очищенный сигнал
        eps (float): Малое значение для избежания деления на ноль

    Возвращает:
        SNR (float): Значение SNR в децибелах
    """
    noise = clean - enhanced
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + eps))

def compute_si_sdr(clean, enhanced, eps=1e-8):
    """
    Вычисляет SI-SDR (Scale-Invariant Signal-to-Distortion Ratio).

    Параметры:
        clean (np.ndarray): Исходный чистый сигнал
        enhanced (np.ndarray): Очищенный сигнал
        eps (float): Малое значение для избежания деления на ноль

    Возвращает:
        SI-SDR (float): Значение SI-SDR в децибелах
    """
    ref_energy = np.sum(clean**2) + eps
    scale = np.sum(clean * enhanced) / ref_energy
    scaled_clean = scale * clean
    noise = scaled_clean - enhanced
    return 10 * np.log10(np.sum(scaled_clean**2) / (np.sum(noise**2) + eps))

def compute_segment_len(T8, n_fft, hop_length):
    """
    Вычисляет длину сегмента аудиосигнала для заданного количества кадров спектрограммы.

    Параметры:
        T8 (int): Количество временных кадров в спектрограмме
        n_fft (int): Размер окна FFT
        hop_length (int): Шаг между окнами

    Возвращает:
        len (int): Длина сегмента аудиосигнала в сэмплах
    """
    return (T8 - 1) * hop_length + n_fft

def extract_phase_and_chunk(noisy_wav, pred_shape, n_fft, hop_length, device):
    """
    Извлекает фазу и сегмент сигнала для реконструкции.

    Параметры:
        noisy_wav (torch.Tensor): Зашумленный аудиосигнал
        pred_shape (tuple): Ожидаемая форма спектрограммы (F, T)
        n_fft (int): Размер окна FFT
        hop_length (int): Шаг между окнами
        device (str): Устройство для вычислений ('cuda' или 'cpu')

    Возвращает:
        signal (tuple): (phase, chunk) — фаза и сегмент сигнала
    """
    F8, T8 = pred_shape
    segment_len = compute_segment_len(T8, n_fft, hop_length)
    chunk = noisy_wav[:, :segment_len]
    window = torch.hann_window(n_fft, device=device)
    spec = torch.stft(
        chunk.to(device),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    phase = torch.angle(spec)[:, :F8, :T8]
    return phase, chunk

def istft_from_mag_phase(pred_mag, noisy_phase, n_fft, hop_length, length):
    """
    Восстанавливает аудиосигнал из магнитуды и фазы.

    Параметры:
        pred_mag (torch.Tensor): Предсказанная магнитуда спектрограммы
        noisy_phase (torch.Tensor): Фаза зашумленного сигнала
        n_fft (int): Размер окна FFT
        hop_length (int): Шаг между окнами
        length (int): Длина выходного сигнала

    Возвращает:
        audio (torch.Tensor): Восстановленный аудиосигнал
    """
    real = pred_mag * torch.cos(noisy_phase)
    imag = pred_mag * torch.sin(noisy_phase)
    spec_ri = torch.stack([real, imag], dim=-1)
    complex_spec = torch.view_as_complex(spec_ri)
    
    # Дополнение по частоте
    n_freq = n_fft // 2 + 1
    if complex_spec.size(1) < n_freq:
        pad = n_freq - complex_spec.size(1)
        zeros = torch.zeros(
            (complex_spec.size(0), pad, complex_spec.size(2)),
            dtype=complex_spec.dtype,
            device=complex_spec.device
        )
        complex_spec = torch.cat([complex_spec, zeros], dim=1)
    
    window = torch.hann_window(n_fft, device=complex_spec.device)
    return torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )

def trainer(num_epochs, batch_size, model_class, criterion, optimizer_class,
            optimizer_params, train_loader, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu',
            n_fft=512, hop_length=128, save_models=False, optimizer_name=None, test=False):
    """
    Функция для обучения модели.

    Параметры:
        num_epochs (int): Количество эпох обучения
        batch_size (int): Размер батча
        model_class (nn.Module): Класс модели
        criterion (nn.Module): Функция потерь
        optimizer_class (optim.Optimizer): Класс оптимизатора
        optimizer_params (dict): Параметры оптимизатора
        train_loader (DataLoader): DataLoader для обучения
        test_loader (DataLoader): DataLoader для тестирования
        device (str): Устройство для вычислений
        n_fft (int): Размер окна FFT
        hop_length (int): Шаг между окнами
        save_models (bool): Флаг сохранения моделей
        optimizer_name (str): Имя оптимизатора (для сохранения)

    Возвращает:
        loss (tuple): (train_losses, test_losses, test_snr, test_si_sdr, model)
    """
    # Инициализация модели и оптимизатора
    model = model_class().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    train_losses = []
    test_losses = []
    test_snr = []
    test_si_sdr = []

    def train_epoch(epoch):
        """
        Одна эпоха обучения.
        """
        model.train()
        epoch_loss = 0
        total_batches = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Эпоха {epoch+1}/{num_epochs} - Обучение",
            unit="batch",
            leave=False
        )

        for noisy_mag, clean_mag, noisy_wav, clean_wav in train_bar:
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            optimizer.zero_grad()
            pred_mag = model(noisy_mag)
            loss = criterion(pred_mag, clean_mag)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_losses.append(batch_loss)
            epoch_loss += batch_loss
            total_batches += 1

            avg_loss = epoch_loss / total_batches
            train_bar.set_postfix(train_loss=f"{avg_loss:.4f}")
            if test:
                break

        return epoch_loss / len(train_loader)

    def test_epoch(epoch):
        """
        Одна эпоха тестирования.
        """
        model.eval()
        epoch_loss = 0

        with torch.no_grad():
            test_bar = tqdm(
                test_loader,
                desc=f"Эпоха {epoch + 1}/{num_epochs} - Тестирование",
                unit="batch",
                leave=False
            )

            for noisy_mag, clean_mag, noisy_wav, clean_wav in test_bar:
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)

                pred_mag = model(noisy_mag)
                loss = criterion(pred_mag, clean_mag)
                batch_loss = loss.item()
                test_losses.append(batch_loss)
                epoch_loss += batch_loss

                # Вычисление метрик качества
                phase, noisy_chunk = extract_phase_and_chunk(
                    noisy_wav,
                    pred_mag.shape[1:],
                    n_fft, hop_length,
                    device
                )
                enhanced_chunk = istft_from_mag_phase(
                    pred_mag, phase,
                    n_fft, hop_length,
                    length=noisy_chunk.shape[1]
                )

                clean_chunk = clean_wav[:, :enhanced_chunk.shape[1]]
                clean_np = clean_chunk.numpy().flatten()
                enhanced_np = enhanced_chunk.cpu().numpy().flatten()

                batch_snr = compute_snr(clean_np, enhanced_np)
                batch_si_sdr = compute_si_sdr(clean_np, enhanced_np)

                test_snr.append(batch_snr)
                test_si_sdr.append(batch_si_sdr)

                avg_loss = epoch_loss / (test_bar.n + 1)
                avg_snr = sum(test_snr[-test_bar.n-1:]) / (test_bar.n + 1)
                test_bar.set_postfix(test_loss=f"{avg_loss:.4f}", SNR=f"{avg_snr:.2f}")
                if test:
                    break

        test_loss = epoch_loss / len(test_loader)
        test_snr_val = sum(test_snr[-len(test_loader):]) / len(test_loader)
        test_si_sdr_val = sum(test_si_sdr[-len(test_loader):]) / len(test_loader)

        return test_loss, test_snr_val, test_si_sdr_val

    # Основной цикл обучения
    for epoch in range(num_epochs):
        start_time = time.time()

        # Обучение и тестирование
        train_loss = train_epoch(epoch)
        test_loss, test_snr_val, test_si_sdr_val = test_epoch(epoch)

        # Логирование
        epoch_time = time.time() - start_time
        mins, secs = divmod(epoch_time, 60)
        time_str = f"{int(mins):02d}:{int(secs):02d}"

        print(
            f"Эпоха {epoch + 1}/{num_epochs} | "
            f"Время: {time_str} | "
            f"Потери (обучение): {train_loss:.4f} | "
            f"Потери (тест): {test_loss:.4f} | "
            f"SNR: {test_snr_val:.2f} дБ | "
            f"SI-SDR: {test_si_sdr_val:.2f} дБ"
        )
    
    # Сохранение модели
    if save_models:
        os.makedirs("models", exist_ok=True)
        model_dict = {
            f'{optimizer_name}': model.state_dict()
        }
        torch.save(model_dict, f'models/{optimizer_name}_denoiser.pt')
        print(f"Модель {optimizer_name} сохранена в models/{optimizer_name}_denoiser.pt")

    return train_losses, test_losses, test_snr, test_si_sdr, model

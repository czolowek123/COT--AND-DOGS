import os
import sys
import time
import cv2
import json
import torch
import logging
import argparse
import inspect
import textwrap
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Any, Union
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image, ImageFile, ImageDraw, ImageFont

try:
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


ImageFile.LOAD_TRUNCATED_IMAGES = True

class GlobalConfig:
    IMG_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 15
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG_DIR = "system_logs"
    MODEL_DIR = "saved_models"
    EXPORT_DIR = "reports"
    
    CAT_DOG_MODEL = "classification_best.pth"
    MASK_MODEL = "mask_detector.pth"
    CLASSES_FILE = "label_map.json"

    @classmethod
    def initialize_env(cls):
        for folder in [cls.LOG_DIR, cls.MODEL_DIR, cls.EXPORT_DIR]:
            if not os.path.exists(folder):
                os.makedirs(folder)

def setup_master_logger():
    GlobalConfig.initialize_env()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(GlobalConfig.LOG_DIR, f"session_{timestamp}.log")

    logger = logging.getLogger("CV_Master")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(threadName)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

log = setup_master_logger()

class MasterDataset(Dataset):
    def __init__(self, root_path: str, transform: transforms.Compose = None):
        self.root = root_path
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        if not os.path.exists(root_path):
            log.error(f"Путь {root_path} не найден! Проверь расположение папки.")
            return

        exclude = [GlobalConfig.LOG_DIR, GlobalConfig.MODEL_DIR, GlobalConfig.EXPORT_DIR, 
                   '__pycache__', '.ipynb_checkpoints', '.git', '.vscode']

        potential_classes = [d for d in os.listdir(root_path) 
                             if os.path.isdir(os.path.join(root_path, d)) and d not in exclude]
        
        for cls_name in potential_classes:
            cls_folder = os.path.join(root_path, cls_name)
            images = [f for f in os.listdir(cls_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if len(images) > 0:
                self.class_names.append(cls_name)
                current_cls_idx = len(self.class_names) - 1
                for img_name in images:
                    self.samples.append((os.path.join(cls_folder, img_name), current_cls_idx))
        
        self.class_names.sort()
        log.info(f"Датасет инициализирован. Найдено классов: {len(self.class_names)} {self.class_names}")
        log.info(f"Общее количество изображений: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            log.warning(f"Ошибка при чтении {path}: {e}. Пропускаем...")
            # Возвращаем следующий элемент в случае ошибки
            return self.__getitem__((idx + 1) % len(self.samples))

class ModelFactory:
    
    @staticmethod
    def get_classification_model(num_classes: int):
        log.info(f"Создание архитектуры ResNet18 для {num_classes} классов...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        for param in model.parameters():
            param.requires_grad = False
            
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        return model.to(GlobalConfig.DEVICE)

    @staticmethod
    def get_transforms(is_train: bool = True):
        if is_train:
            return transforms.Compose([
                transforms.Resize((GlobalConfig.IMG_SIZE, GlobalConfig.IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((GlobalConfig.IMG_SIZE, GlobalConfig.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

class CoreEngine:
    
    def __init__(self, task_name: str):
        self.task = task_name
        self.history = {'loss': [], 'val_acc': []}

    def train_process(self, dataset_path: str):
        log.info(f"--- ЗАПУСК ОБУЧЕНИЯ ДЛЯ ЗАДАЧИ: {self.task.upper()} ---")

        full_dataset = MasterDataset(dataset_path)
        if len(full_dataset) < 2:
            log.error("КРИТИЧЕСКАЯ ОШИБКА: Недостаточно данных для обучения! Нужно минимум 2 класса по несколько фото.")
            self._print_troubleshooting()
            return

        train_len = int((1 - GlobalConfig.VAL_SPLIT) * len(full_dataset))
        val_len = len(full_dataset) - train_len
        if val_len == 0: train_len, val_len = len(full_dataset)-1, 1

        train_set, val_set = random_split(full_dataset, [train_len, val_len])
        
        train_set.dataset.transform = ModelFactory.get_transforms(is_train=True)
        val_set.dataset.transform = ModelFactory.get_transforms(is_train=False)

        train_loader = DataLoader(train_set, batch_size=GlobalConfig.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=GlobalConfig.BATCH_SIZE)

        model = ModelFactory.get_classification_model(len(full_dataset.class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=GlobalConfig.LEARNING_RATE)

        self._save_labels(full_dataset.class_names)

        best_acc = 0.0
        for epoch in range(GlobalConfig.EPOCHS):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(GlobalConfig.DEVICE), labels.to(GlobalConfig.DEVICE)
                optimizer.zero_grad()
                
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_acc = self._evaluate(model, val_loader)
            self.history['loss'].append(running_loss / len(train_loader))
            self.history['val_acc'].append(val_acc)

            duration = time.time() - t0
            log.info(f"Эпоха [{epoch+1}/{GlobalConfig.EPOCHS}] | Loss: {self.history['loss'][-1]:.4f} | Val Acc: {val_acc:.2f}% | Время: {duration:.1f}с")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(GlobalConfig.MODEL_DIR, GlobalConfig.CAT_DOG_MODEL))
                log.info(">>>> Модель сохранена как лучшая!")

        self._plot_results()
        log.info("Обучение успешно завершено.")

    def _evaluate(self, model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(GlobalConfig.DEVICE), labels.to(GlobalConfig.DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def _save_labels(self, class_list):
        with open(os.path.join(GlobalConfig.MODEL_DIR, GlobalConfig.CLASSES_FILE), 'w') as f:
            json.dump(class_list, f)

    def _plot_results(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], color='red', lw=2)
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_acc'], color='blue', lw=2)
        plt.title('Validation Accuracy')
        plt.grid(True)
        
        plt.savefig(os.path.join(GlobalConfig.EXPORT_DIR, 'learning_curves.png'))
        plt.close()

    def _print_troubleshooting(self):
        msg = """
        [!] СОВЕТ ПО ИСПРАВЛЕНИЮ:
        1. Создай папку 'data'.
        2. Внутри 'data' создай папки 'Cat' и 'Dog'.
        3. Положи минимум по 5 картинок в каждую.
        4. Запусти команду: python main.py --task cats_dogs --mode train --dataset ./data
        """
        print(textwrap.dedent(msg))

class InferenceUnit:

    @staticmethod
    def classify_image(img_path: str):
        log.info(f"Анализ изображения: {img_path}")
        
        try:
            with open(os.path.join(GlobalConfig.MODEL_DIR, GlobalConfig.CLASSES_FILE), 'r') as f:
                classes = json.load(f)
        except:
            log.error("Файл меток не найден! Сначала обучи модель.")
            return

        model = ModelFactory.get_classification_model(len(classes))
        model.load_state_dict(torch.load(os.path.join(GlobalConfig.MODEL_DIR, GlobalConfig.CAT_DOG_MODEL), map_location=GlobalConfig.DEVICE))
        model.eval()

        img_pil = Image.open(img_path).convert('RGB')
        transform = ModelFactory.get_transforms(is_train=False)
        tensor = transform(img_pil).unsqueeze(0).to(GlobalConfig.DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)

        res_text = f"Результат: {classes[pred.item()]} ({conf.item()*100:.1f}%)"
        log.info(res_text)

        plt.imshow(img_pil)
        plt.title(res_text)
        plt.axis('off')
        plt.show()

    @staticmethod
    def license_plate_ocr(img_path: str):
        log.info(f"Запуск OCR-модуля для: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            log.error("Не удалось открыть файл изображения!")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise_removed = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(noise_removed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        reader = easyocr.Reader(['en', 'ru'], gpu=torch.cuda.is_available())
        results = reader.readtext(thresh)

        log.info(f"Найдено текстовых блоков: {len(results)}")
        for (bbox, text, prob) in results:
            log.info(f" >> Текст: '{text}' (Уверенность: {prob:.2f})")
            
            p1 = tuple(map(int, bbox[0]))
            p2 = tuple(map(int, bbox[2]))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, text, (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("OCR Result (Press any key)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="AI Computer Vision Tool v3.0")
    parser.add_argument('--task', type=str, required=True, choices=['cats_dogs', 'ocr', 'masks'], 
                        help="Выбор задачи: классификация, текст или маски")
    parser.add_argument('--mode', type=str, default='inference', choices=['train', 'inference'],
                        help="Режим: обучение или проверка")
    parser.add_argument('--dataset', type=str, default='.', help="Путь к папке с данными")
    parser.add_argument('--image', type=str, default='test.jpg', help="Путь к картинке для проверки")
    
    args = parser.parse_args()

    GlobalConfig.initialize_env()
    
    log.info("="*50)
    log.info("ЗАПУСК СИСТЕМЫ AI VISION")
    log.info(f"Устройство: {GlobalConfig.DEVICE}")
    log.info("="*50)

    try:
        if args.task == 'cats_dogs':
            engine = CoreEngine("Classification")
            if args.mode == 'train':
                engine.train_process(args.dataset)
            else:
                InferenceUnit.classify_image(args.image)

        elif args.task == 'ocr':
            InferenceUnit.license_plate_ocr(args.image)

        elif args.task == 'masks':
            if args.mode == 'train':
                engine = CoreEngine("MaskDetection")
                engine.train_process(args.dataset)
            else:
                log.info("Используйте классификацию с весами масок для проверки.")

    except KeyboardInterrupt:
        log.warning("Процесс прерван пользователем.")
    except Exception as e:
        log.error(f"Критическая ошибка выполнения: {e}", exc_info=True)
    finally:
        log.info("Сессия завершена.")

if __name__ == "__main__":
    main()
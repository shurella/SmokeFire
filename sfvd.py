import cv2
import numpy as np
from collections import deque

# Настройки параметров
FIRE_HSV_MIN = (0, 100, 100)    # Нижний порог огня (H,S,V)
FIRE_HSV_MAX = (20, 255, 255)    # Верхний порог огня
SMOKE_HSV_MIN = (0, 0, 200)      # Нижний порог дыма
SMOKE_HSV_MAX = (180, 50, 255)   # Верхний порог дыма
MIN_AREA = 500                   # Минимальная площадь контура
SENSITIVITY = 1.0                # Чувствительность (0-1)
HISTORY_SIZE = 10                # Глубина истории для анализа движения

# Инициализация
cap = cv2.VideoCapture(0)        # Или укажите путь к видео
kernel = np.ones((5,5), np.uint8)
motion_history = deque(maxlen=HISTORY_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Предварительная обработка
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 2. Создание масок с регулируемой чувствительностью
    fire_mask = cv2.inRange(hsv, 
        tuple(int(SENSITIVITY * x) for x in FIRE_HSV_MIN),
        tuple(min(255, int((2-SENSITIVITY) * x)) for x in FIRE_HSV_MAX))
    
    smoke_mask = cv2.inRange(hsv, 
        tuple(int(SENSITIVITY * x) for x in SMOKE_HSV_MIN),
        tuple(min(255, int((2-SENSITIVITY) * x)) for x in SMOKE_HSV_MAX))
    
    # 3. Морфологические операции
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Анализ движения для дыма
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_history.append(gray)
    
    if len(motion_history) == HISTORY_SIZE:
        motion_diff = cv2.absdiff(motion_history[0], motion_history[-1])
        _, motion_mask = cv2.threshold(motion_diff, 25, 255, cv2.THRESH_BINARY)
        smoke_mask = cv2.bitwise_and(smoke_mask, motion_mask)
    
    # 5. Поиск контуров
    fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Отрисовка результатов
    for cnt in fire_contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, f'Fire {area:.0f}px', (x,y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    for cnt in smoke_contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (200,200,200), 2)
            cv2.putText(frame, f'Smoke {area:.0f}px', (x,y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    
    # 7. Вывод информации
    cv2.putText(frame, f'Sensitivity: {SENSITIVITY:.1f}', (10,30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # 8. Отображение
    cv2.imshow('Fire/Smoke Detection', frame)
    #cv2.imshow('Fire Mask', fire_mask)
    #cv2.imshow('Smoke Mask', smoke_mask)
    
    # Управление с клавиатуры
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        SENSITIVITY = min(1.0, SENSITIVITY + 0.1)
    elif key == ord('-'):
        SENSITIVITY = max(0.1, SENSITIVITY - 0.1)

cap.release()
cv2.destroyAllWindows()

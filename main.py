import cv2
import numpy as np
from ligne import show_angle

# Image de fond
filename="part3.png"
image = cv2.imread(filename)
if image is None:
    raise FileNotFoundError(f"the file  {filename} is not found.")

# Initialisation
points = []  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
selected_point_idx = None
dragging = False
radius = 6

# Détection de point proche
def find_nearest_point(x, y):
    for i, (px, py) in enumerate(points):
        if np.hypot(px - x, py - y) < radius + 3:
            return i
    return None

# Callback souris
def mouse_event(event, x, y, flags, param):
    global selected_point_idx, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = find_nearest_point(x, y)
        if idx is not None:
            selected_point_idx = idx
            dragging = True
        elif len(points) < 4:
            points.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_point_idx is not None:
        points[selected_point_idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_point_idx = None
    
    
cv2.namedWindow("Angles detection ")
cv2.setMouseCallback("Angles detection ", mouse_event)

print("\033[92m🖱 Draw two lines on the angle measurement areas. Press ESC to exit.\033[0m")
while True:
    img_disp = image.copy()

   
   
    if len(points) == 4:

        # Affichage de l'angle entre les deux lignes
        show_angle([[points[0][0], points[0][1], points[1][0], points[1][1]],
                    [points[2][0], points[2][1], points[3][0], points[3][1]]],
                   img_disp)
        cv2.line(img_disp, points[2], points[3], (0, 255, 0), 2)
    if len(points) >= 2:
        cv2.line(img_disp, points[0], points[1], (255, 0, 0), 2)
    for pt in points:
        cv2.circle(img_disp, pt, radius, (0, 0, 255), -1)

    # Afficher les coordonnées
    for i, (x, y) in enumerate(points):
        cv2.putText(img_disp, f"P{i}:({x},{y})", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    cv2.imshow("Angles detection ", img_disp)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# Résultat final
if len(points) == 4:
    line1 = (*points[0], *points[1])
    line2 = (*points[2], *points[3])
    print("Line 1 :", line1)
    print("Line 2 :", line2)
else:
    print("You have not defined both complete lines..")
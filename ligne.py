import cv2
import numpy as np
import math




def intersection_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calcul des dénominateurs
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # Lignes parallèles (ou colinéaires)

    # Calcul des coordonnées d'intersection
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return int(px), int(py)


def affichage_angle_vecteurs(image, line1, line2, couleur=(0, 0, 255), rayon=40):
    # Trouver le point d’intersection
    inter = intersection_lines(line1, line2)
    if inter is None:
        return image  # rien à faire si pas d'intersection

    x, y = inter

    # Vecteurs directionnels depuis l’intersection
    def vecteur_depuis(p1, p2):
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        return v / np.linalg.norm(v)

    v1 = vecteur_depuis(inter, (line1[2], line1[3]))
    v2 = vecteur_depuis(inter, (line2[2], line2[3]))

    end1 = (int(x + v1[0]*rayon), int(y + v1[1]*rayon))
    end2 = (int(x + v2[0]*rayon), int(y + v2[1]*rayon))

    # Tracer les flèches
    cv2.arrowedLine(image, (x, y), end1, (255, 0, 0), 2, tipLength=0.2)
    cv2.arrowedLine(image, (x, y), end2, (0, 255, 0), 2, tipLength=0.2)

    # Angle entre les vecteurs
    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = math.degrees(angle_rad)

    # Dessiner l’arc
    axes = (rayon, rayon)
    angle1 = math.degrees(math.atan2(-v1[1], v1[0])) % 360
    angle2 = math.degrees(math.atan2(-v2[1], v2[0])) % 360
    span = (angle2 - angle1) % 360
    cv2.ellipse(image, (x, y), axes, 0, angle1, angle1 + span, couleur, 2)

    # Texte de l’angle
    cv2.putText(image, f"{angle_deg:.2f}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    return image
def distance_between_parallel_lines(line1, line2):
    # line1 = (x1, y1, x2, y2)
    # line2 = (x3, y3, x4, y4)
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Vecteur directeur de la 2ème ligne
    dx = x4 - x3
    dy = y4 - y3

    # Normaliser le vecteur
    norm = np.hypot(dx, dy)
    if norm == 0:
        return None  # ligne 2 invalide
    dx /= norm
    dy /= norm

    # Vecteur du point (x1, y1) à (x3, y3)
    vx = x1 - x3
    vy = y1 - y3

    # Produit vectoriel pour trouver la distance orthogonale
    distance = abs(vx * dy - vy * dx)
    return distance
def distance_entre_deux_lignes(line1, line2):
    def point_to_segment_distance(px, py, x1, y1, x2, y2):
        # vecteur segment
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:
            return np.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.hypot(px - proj_x, py - proj_y)

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculer les distances point-segment dans les deux sens
    d1 = point_to_segment_distance(x1, y1, x3, y3, x4, y4)
    d2 = point_to_segment_distance(x2, y2, x3, y3, x4, y4)
    d3 = point_to_segment_distance(x3, y3, x1, y1, x2, y2)
    d4 = point_to_segment_distance(x4, y4, x1, y1, x2, y2)

    return min(d1, d2, d3, d4)
def adjust_line_length(line1, line2):
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    # Longueurs des deux lignes
    len1 = np.linalg.norm([x12 - x11, y12 - y11])
    len2 = np.linalg.norm([x22 - x21, y22 - y21])

    if len1 > len2:
        # Direction de la ligne 2 (vecteur unitaire)
        dx = x22 - x21
        dy = y22 - y21
        norm = np.sqrt(dx**2 + dy**2)
        dx /= norm
        dy /= norm

        # Allonger ligne 2 à la même longueur que ligne 1
        new_x22 = int(x21 + dx * len1)
        new_y22 = int(y21 + dy * len1)
        line2=x21, y21, new_x22, new_y22
        return line1, line2
    else:
        # Si ligne2 est plus longue, on peut faire l'inverse
        dx = x12 - x11
        dy = y12 - y11
        norm = np.sqrt(dx**2 + dy**2)
        dx /= norm
        dy /= norm

        new_x12 = int(x11 + dx * len2)
        new_y12 = int(y11 + dy * len2)
        line1=x11, y11, new_x12, new_y12
        return line1, line2
def moyen_2_lignes(line1,line2):
    line1,line2 =adjust_line_length(line1, line2)
     
    x11,y11,x12,y12=line1 
    x21,y21,x22,y22=line2
    X1=int((x11+x21)/2) 
    X2=int((x22+x12)/2)
    Y1=int((y11+y21)/2)                 
    Y2=int((y12+y22)/2)                 
   
    line_moyen=X1,Y1,X2,Y2
   
    return line_moyen
def aligne_au_plus_petit_point(line1, line2):
    # Décomposer les lignes
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    # Trouver le point de départ de chaque ligne (celui avec x + y le plus petit)
    start1 = (x11, y11) if (x11 + y11) < (x21 + y21) else (x21, y21)
    start2 = (x12, y12) if (x12 + y12) < (x22 + y22) else (x22, y22)

    # Trouver le point le plus "haut à gauche"
    x_base, y_base = start1 if (start1[0] + start1[1]) < (start2[0] + start2[1]) else start2

    # Recalculer la direction de chaque ligne
    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21

    # Redéfinir chaque ligne en partant du point de base avec son propre vecteur
    new_line1 = (x_base, y_base, x_base + dx1, y_base + dy1)
    new_line2 = (x_base, y_base, x_base + dx2, y_base + dy2)

    return new_line1, new_line2
def calcule_angle(line1,line2):
    x11, y11, x12, y12=line1
    x21, y21, x22, y22=line2
    v1 = np.array([x12 - x11, y12 - y11])
    v2 = np.array([x22 - x21, y22 - y21])

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return  # ligne trop courte ou erreur

    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def Lines_hough(lines):
 line_fus = []
 skip_indices = set()
 skip=False   
 lines1=lines
 lines2=lines
 k=0
 if lines1 is not None:
    for i in range(len(lines1)):
        skip=False
    
        
        x11, y11, x12, y12 = lines1[i][0]
        line1=x11, y11, x12, y12
        for j in range(i , len(lines2)):
            if j in skip_indices:
                continue
            x21, y21, x22, y22 = lines2[j][0]
            line2=x21, y21, x22, y22
            if distance_between_parallel_lines(line1,line2)>10 and distance_entre_deux_lignes(line1, line2)>30 :
                continue
            angle_deg=calcule_angle(line1,line2)
            angle_deg = min(angle_deg, 180 - angle_deg)
            if angle_deg < 5 :  # ≈ lignes parallèles
              skip_indices.add(j)
              if skip==False :  
                line_fus.append(moyen_2_lignes(line1,line2))
                k=k+1
              else :
                line_fus[k-1]=moyen_2_lignes(line_fus[k-1],line2)
              skip=True

            elif j==len(lines2) and skip==False :
                line_fus.append(line1)
                k=k+1
                
    return line_fus   
            
    
            

    
def show_angle(CORR,image):
     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Corrigé: BGR et non RGB
     blurred = cv2.GaussianBlur(image_gray, (9, 9), 0)
     edges = cv2.Canny(blurred, 50, 200)
     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=1, minLineLength=100, maxLineGap=10)  
     line_fus=Lines_hough(lines) 
     t=0
     print(len(CORR))   
     if len(CORR)>=2 :
        LCORR = np.zeros((4, 4), dtype=int)
        print(LCORR)
        ligne_Corr=set()
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            t=t+1
            
        for i in range(len(line_fus)):
            x1, y1, x2, y2 = line_fus[i]
            for j in range(len(CORR)):
                coef=np.corrcoef(line_fus[i],CORR[j])[0,1]
                angle=calcule_angle(line_fus[i],CORR[j])
                
                dist=distance_entre_deux_lignes(line_fus[i], CORR[j])
                print(f"Anlges {angle} - Distance {dist} - Correlation {coef}" )
                if (0<angle< 10 or 170<angle<180)  and dist<45:
                    print(f"Anlges {angle}")
                    print(f"distance{dist}")
                    LCORR[j]=line_fus[i]
                    ligne_Corr.add(i)
        for i in range (len(line_fus)):
            x1, y1, x2, y2 = line_fus[i]
            if i not in ligne_Corr:
              cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            else :
              cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)               
        for i in range(len(LCORR)-1):
            affichage_angle_vecteurs(image, LCORR[i],LCORR[i+1], couleur=(0, 0, 255), rayon=40)
     else :
         print("Longeur CORR < 2")
       
    


           
        
     

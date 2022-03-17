# Einfluss von Buntheit auf den wahrgenommenen Realismus rekolorierter Bilder

## Erdem Arslan, Savas Großmann, Marius Krause, Max Mühlefeldt

### Seminar: Visuelle Wahrnehmung beim Menschen und Bildqualität - WiSe 2021/22

## 1. Einleitung

Historische, schwarz-weiß Fotografien können unter anderem mittels Machine Learning eingefärbt werden (rekoloriert). Wir untersuchen die Anpassung der Buntheit der eingefärbten Bilder. Die vorliegende Fragestellung ist, welchen Einfluss die Buntheit auf den wahrgenommenen Realismus rekolorierter Bilder hat.

Zur Untersuchung der Fragestellung wurden historische und moderne Fotografien rekoloriert und nachträglich in ihrerer Buntheit manipuliert.

Wir haben die Hypothese untersucht, dass ein bunteres, rekoloriertes Bild, als realistischer wahrgenommen wird.

### Buntheit
Basierend auf DIN EN ISO/CIE 11664-4 werden die Bilder in den CIELAB Farbraum konvertiert. Im CIELAB Farbraum ist die Buntheit von jedem Bildpunkt gut manipulierbar. Für jeden Bildpunkt werden die folgenden Informationen gespeichert:
* L*: Helligkeit
* a*: Rot-Grün-Buntheit
* b*: Gelb-Blau-Buntheit

Die Buntheit kann mittels `C*_ab = sqrt(a*² b*²)` für jeden Bildpunkt errechnet werden.



Farbraumkonvertierungen wurden mittels des `skimage` Pakets für Python umgesetzt.

```
from skimage import io, color
original_color_lab_image = color.rgb2lab(io.imread(file_path))
```

### Machine Learning Modell
Zum Einsatz kommt hier ein bereits trainiertes Machine Learning Modell. Das Modell wurde duch Zhang et al. entwickelt [1]. Zu beachten ist, dass das Modell nur mittels moderner Bilder trainiert wurde. Im Training wurde ein Bild im CIELAB Farbraum, dessen Buntheitswerte a* und b* entfernt wurden, als Eingabe für das Modell verwendet. Der Output des Modells wurde mit dem originalen Bild, samt Buntheitswerten, verglichen. Basierend auf dem Vergleich erfolgte das Training des Modells.

Mit dieser Methode ist es nicht möglich das Modell mit den historischen Bildern zu trainieren, da ein Vergleich mit existierenden Buntsheitswerten nicht möglich ist.


### Gewählte Bilder


## 2. Experimentelles Design



## 3. Ergebnisse




### Qualitative Beobachtungen



## 4. Diskussion



### Mögliche Probleme 


### Offene Fragen

### Referenzen
[1] Zhang et al. Colorful Image Colorization, ECCV Proceedings, 2016.


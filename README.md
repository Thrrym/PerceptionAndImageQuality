# Einfluss von Buntheit auf den wahrgenommenen Realismus rekolorierter Bilder

## Erdem Arslan, Savas Großmann, Marius Krause, Max Mühlefeldt

### Seminar: Visuelle Wahrnehmung beim Menschen und Bildqualität - WiSe 2021/22

## 1. Einleitung

Historische Schwarz-Weiß-Fotografien können unter anderem mittels Machine Learning eingefärbt werden (rekoloriert). Wir untersuchen den Einfluss der Buntheit der eingefärbten Bilder auf den wahrgenommenen Realismus. Die vorliegende Fragestellung ist, welchen Einfluss die Buntheit auf den wahrgenommenen Realismus rekolorierter Bilder hat.

Zur Untersuchung der Fragestellung wurden historische und moderne Fotografien rekoloriert und nachträglich in ihrerer Buntheit manipuliert.

Wir haben die Hypothese untersucht, dass ein bunteres, rekoloriertes Bild, als realistischer wahrgenommen wird.

### Buntheit
Zentraler Begriff der vorliegenden Untersuchung ist Buntheit bzw. Chroma.

Basierend auf DIN EN ISO/CIE 11664-4 werden die Bilder in den CIELAB Farbraum konvertiert. Im CIELAB Farbraum ist die Buntheit von jedem Bildpunkt berechen- und manipulierbar. Für jeden Bildpunkt werden im CIELAB Frabraum die folgenden Informationen gespeichert:
* L*: Helligkeit
* a*: Rot-Grün-Buntheit
* b*: Gelb-Blau-Buntheit

Die Buntheit kann mittels `C*_ab = sqrt((a*)² + (b*)²)` für jeden Bildpunkt errechnet werden. Die Buntheit eines Bildpunktes ist somit von a* und b* abhängig. Eine einfache Darstellung des Zusammhangs von a*, b* und der Buntheit im CIELAB Farbraums kann der folgenden Darstellungen entnommen werden [4]:

![CIELAB](img_lab.png)

Farbraumkonvertierungen von sRGB zu CIELAB und umgekehrt wurden mittels des `skimage` Pakets für Python umgesetzt [2]:

```python
from skimage import io, color
original_color_lab_image = color.rgb2lab(io.imread(file_path))
```

Die Manipulation der Buntheit eines Bildes im CIELAB Farbraum erfolgt durch elementweise Matrixmultiplikation. Die Einträge für einen Bildpunkt werden mit dem Vektor [1, f, f], mit f als Faktor, um den die Buntheit angepasst wird:
* L*: Bleibt durch Multiplikation mit 1 unverändert.
* a* und b*: Beide Einträge werden durch die elementweise Multiplikation mit f angepasst.

Eine entsprechende Matrix für das gesamte Bild kann wie folgt erstellt werden.

```python
def get_modification_matrix(chroma_factor, requested_shape):
    vector_to_multiply_elementwise = np.array([1, chroma_factor, chroma_factor])
    return np.tile(
        vector_to_multiply_elementwise[None][None], (requested_shape[0], requested_shape[1], 1)
    )
```

Die elementweise Multiplikation erfolgt mittels der Funktion

```python
def modify_lab_image_chroma(chroma_factor, image_to_modify):
    """ Modify the provided image in Lab color space by given chroma factor. """
    # Generate the modifier matrix.
    modifier = get_modification_matrix(chroma_factor, image_to_modify.shape)

    # Elementwise multiplication of original image.
    modified_image = image_to_modify * modifier
```

#### Hinweis zum CIELAB Farbraum
Die Buntheit im CIELAB Farbraum kann beliebig angepasst werden. Jedoch ist ein Konvertierung dieser Werte in einen anderen Farbraum nur noch bedingt möglich. Gerade für die Darstellung auf Computerbildschirmen ist eine Konvertierung in den sRGB-Farbraum notwendig. Entsprechend sieht auch das `skimage` Paket Limitierungen für die Konvertierung aus dem CIELAB Farbraum vor. So müssen a* und b* im Intervall [0, 100] in den reellen Zahlen liegen.

### Gewählte Bilder
Es wurden Bilder in zwei Gruppen gewählt: Historische Bilder und moderne Bilder.

#### Historische Bilder
Historische Bilder aus verschiedenen Quellen wurden zusammengetragen. Es wurden auf eine vielfältige 

#### Moderne Bilder
Die Modernen Bilder stellen eine kleinere Kontrollgruppe da. Mit diesen Bildern wurde der verwendete Machine Learning Algorithmus trainiert [1]. Die Bilder wurden der Tampere Image Database 2013 entnommen [3]. Die Bilder wurden aus thematischen Themen gewählt, die auch in der Gruppe der historischen Bildern zum Einsatz kommt.


### Machine Learning Modell
Zum Einsatz kommt hier ein bereits trainiertes Machine Learning Modell. Das Modell wurde duch Zhang et al. entwickelt [1]. Zu beachten ist, dass das Modell nur mittels moderner Bilder trainiert wurde. Im Training wurde ein Bild im CIELAB Farbraum, dessen Buntheitswerte a* und b* entfernt wurden, als Eingabe für das Modell verwendet. Der Output des Modells wurde mit dem originalen Bild, samt Buntheitswerten, verglichen. Basierend auf dem Vergleich erfolgte das Training des Modells.

Mit dieser Methode ist es nicht möglich das Modell mit den historischen Bildern zu trainieren, da ein Vergleich mit existierenden Buntsheitswerten nicht möglich ist.




## 2. Vorbereitung Stimuli
Wir beschreiben die Erstellung der Stimuli und beziehen uns auf die Ordner in `code/image_generation`.

* `00_base_images`
* `01_conversion_modern_images_to_to_bw`
* `02_recolor`
* `03_modify_chroma`
* `04_completed_images`

## 2. Experimentelles Design



## 3. Ergebnisse




### Qualitative Beobachtungen



## 4. Diskussion



### Mögliche Probleme 


### Offene Fragen

### Referenzen
[1] Zhang et al. Colorful Image Colorization, ECCV Proceedings, 2016.

[2] scikit-image development team. Besucht am 21.03.2022, https://scikit-image.org/.

[3] Tampere Image Database. Besucht am 13.12.2021, https://www.ponomarenko.info/tid2013.htm.

[4] Konica Minolta. Precise Color Communication. Besucht am 08.12.2021, https://www.konicaminolta.com/instruments/download/booklet/index.html.
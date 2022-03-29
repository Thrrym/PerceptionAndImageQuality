# Einfluss von Buntheit auf den wahrgenommenen Realismus rekolorierter Bilder

## Erdem Arslan, Savas Großmann, Marius Krause, Max Mühlefeldt

### Seminar: Visuelle Wahrnehmung beim Menschen und Bildqualität - WiSe 2021/22

## Inhaltsverzeichnis

[1. Einleitung](#1-einleitung)

[2. Vorbereitung der Stimuli](#2-vorbereitung-der-stimuli)

[3. Experimentelles Design](#3-experimentelles-design)

[4. Ergebnisse](#4-ergebnisse)

[5. Diskussion](#5-diskussion)

## 1. Einleitung

Historische Schwarz-Weiß-Fotografien können mittels Machine Learning Algorithmen nachträglich eingefärbt bzw. rekoloriert werden. Wir untersuchen den Einfluss der Buntheit der eingefärbten Bilder auf den wahrgenommenen Realismus. Die vorliegende Fragestellung ist, welchen Einfluss die Buntheit auf den wahrgenommenen Realismus der rekolorierter Bilder hat. Als Hypothese wird angenommen, dass ein bunteres, nachträglich eingefärbtes Bild als realistischer wahrgenommen wird.

Die Untersuchung erfolgt anhand von historischen und modernen Fotografien. Diese werden rekoloriert und neun Varianten unterschiedlicher Buntheit erstellt. Beobachtern wurde im Anschluss die Varianten der Bilder gezeigt und nach der "realistischsten" Variante gefragt.

### Buntheit

Wir gehen zunächst auf den Begriff der Buntheit und den CIELAB Farbraum ein, der unter anderem die Manipulation der Buntheit erlaubt. 

#### Allgemein

Zentraler Begriff der vorliegenden Untersuchung ist Buntheit bzw. Chroma. Hier wird Buntheit als Anteil von Schwarz __und__ Weiß in einer Farbe verstanden. Je mehr Schwarz- und Weißanteil in einer Farbe enthalten ist, desto geringer ist die Buntheit einer Farbe. Verdeutlicht wird dies und der Unterscheid zur Sättigung (nur Weißanteil) im Farbtongleichen Dreieck [5]:

![Farbtongleiches Dreieck](img_farb_dreieck.png)

#### CIELAB Farbraum

Basierend auf DIN EN ISO/CIE 11664-4 werden die Bilder in den CIELAB oder L* a* b* Farbraum konvertiert. Im CIELAB Farbraum ist die Buntheit von jeder Farbe bzw. Bildpunkt berechen- und manipulierbar. Für jeden Bildpunkt werden im CIELAB Frabraum die folgenden Informationen gespeichert:

* L*: Helligkeit
* a*: Rot-Grün-Buntheit
* b*: Gelb-Blau-Buntheit

Die Buntheit kann mittels der Formel 

![Cab Formel](img_cab_formel.png)

für jeden Bildpunkt errechnet werden. Die Buntheit eines Bildpunktes ist somit von a* und b* abhängig. Eine einfache Darstellung des Zusammhangs von a*, b* und der Buntheit im CIELAB Farbraum kann der folgenden Darstellungen entnommen werden [4]:

![CIELAB](img_lab.png)

Die Abbildung zeigt, dass eine Änderung der Buntheit (Chroma in der Abb.) den  Farbton nicht beeinflusst (Hue in der Abb.). Wird die Buntheit eines Bildpunktes geändert, hat das keinen Einfluss auf den Farbton des Bildpunktes.

Farbraumkonvertierungen vom sRGB- zum CIELAB Farbraum und umgekehrt wurden mittels des `skimage` Pakets für Python 3 umgesetzt [2]. Hier die Umwandlung sRGB zu CIELAB:

```python
from skimage import io, color
# Read image from path and perform conversion.

lab_image = color.rgb2lab(io.imread(file_path))
```

Die Manipulation der Buntheit eines Bildes im CIELAB Farbraum erfolgt durch elementweise Matrixmultiplikation. Die Einträge für einen Bildpunkt werden mit dem Vektor `[1, f, f]` elementweise multipliziert. Dabei ist f der Faktor, um den die Buntheit angepasst wird:
* L* (Helligkeit): Bleibt durch Multiplikation mit 1 unverändert.
* a* und b*: Beide Einträge werden durch die elementweise Multiplikation mit f angepasst.

Eine entsprechende Matrix für das gesamte Bild kann wie folgt erstellt werden.

```python
import numpy as np

def get_modification_matrix(chroma_factor, requested_shape):
    """ Get matrix for element wise multiplication in the request shape.
    Usually the shape is equal to the shape of the original image
    as provided by the ML algorithm. """

    # Get the vector [1, f, f] with f as mod. factor.
    vector_to_multiply_elementwise = np.array([1, chroma_factor, chroma_factor])

    # Return matrix in the desired shape.
    return np.tile(
        vector_to_multiply_elementwise[None][None], (requested_shape[0], requested_shape[1], 1)
    )
```

Die elementweise Multiplikation erfolgt mittels der Funktion `modify_lab_image_chroma()`:

```python
def modify_lab_image_chroma(chroma_factor, image_to_modify):
    """ Modify the provided image in Lab color space by given chroma factor. """
    # Generate the modifier matrix.
    modifier = get_modification_matrix(chroma_factor, image_to_modify.shape)

    # Elementwise multiplication of original image.
    modified_image = image_to_modify * modifier
```

#### Hinweis zum CIELAB Farbraum
Die Buntheit im CIELAB Farbraum kann beliebig angepasst werden. Jedoch ist ein Konvertierung dieser Werte in einen anderen Farbraum nur noch bedingt möglich. Gerade für die Darstellung auf Computerbildschirmen ist eine Konvertierung in den sRGB-Farbraum notwendig. Entsprechend sieht auch das `skimage` Paket Limitierungen für die Konvertierung aus dem CIELAB Farbraum vor. So müssen a* und b* im Intervall [-100, 100] in den reellen Zahlen liegen. Es ergibt sich somit auch, dass die Buntheit im Intervall [0, 100] liegt.

#### Wahl der Buntheitsfaktoren
Basierend auf den oben gezeigten Grenzen für die Manipulation der Buntheit sind die Faktoren für die Manipulation der Buntheit zu wählen. Die Varianten eines Bildes mit unterschiedlichen Buntheiten enthält in der vorliegenden Untersuchung immer:
* Faktor 1,0: Buntheit des Bildes, wie durch den Algorithmus generiert (Original rekoloriertes Bild).
* Faktor 0,6: Buntheit reduziert um Faktor 0,6. Hintergrund ist, dass sichergestellt wird, das den Beobachtern auch immer eine weniger bunte Variante gezeigt wird, als durch den Algorithmus generiert wird. Es wäre möglich, dass der Algorithmus grundsätzlich zu bunte Bilder generiert.
* Faktor max: Basierend auf der höchsten Buntheit des gesamten original rekolorierten Bildes, der höchst mögliche Buntheitswert. Wie durch die folgende Funktion bestimmt:
```python
import numpy as np
def get_max_chroma_factor(image_to_modify):
    """Calculate max possible Chroma value."""
    min_a = np.abs(np.amin(image_to_modify[:, :, 1]))
    max_a = np.abs(np.amax(image_to_modify[:, :, 1]))
    min_b = np.abs(np.amin(image_to_modify[:, :, 2]))
    max_b = np.abs(np.amax(image_to_modify[:, :, 2]))
    max_abs_value = max(min_a, max_a, min_b, max_b)
    ab_abs_max = 99
    return ab_abs_max / max_abs_value
```
* Die verbleibenden sieben Faktoren: Die weiteren Faktoren werden mittels Interpolation bestimmt. Die Bestimmung der Buntheitsfaktoren eines Bildes erfolgt in `03_modify_chroma/main.py` mittels Funktion `get_factors_between_min_max()` mit Eingabeparametern `min_value=0.6` für den kleinsten Faktor und `max_value` der berechnete maximal mögliche Buntheitswert:

```python
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
import numpy as np

def get_factors_between_min_max(min_value, max_value) -> list:
    """Calculate chroma manipulation factors between min, 1.0 and max value."""
    # 1. step: Get linear function between (min_value, 0) and (max_value, 8).
    # Background: Needed to calculate the position of the factor == 1.0
    def fit_func(x, a, b):
        return a*x + b
    popt, pcov = curve_fit(fit_func, [min_value, max_value], [0, 8])

    # 2. step: Using three known factors (min, 1.0, max) fit interpolated function.
    
    # Get number of factor == 1. Needed to know position within the 9 factors.
    # Example: Worst case max factor == 1 -> factor == 1 is the factor with index 8.
    factor_num_1 = int(fit_func(1.0, *popt))
    if factor_num_1 < 1.0:
        factor_num_1 = 1.0
    fixed_factors_y = [min_value, 1.0, max_value]
    fixed_factors_x = [0, factor_num_1, 8]
    calculated_function = Akima1DInterpolator(fixed_factors_x, fixed_factors_y)

    # 3. step: Calculate nine chroma factors:
    x_range = np.arange(9)
    return list(calculated_function(x_range))
```

Im Ergbnis erhalten wir Buntkeitsfaktoren für jedes Bild mit festen Faktoren, hier am Beispiel für Bild `I04`:

![Faktoren](I04_chroma_factors.png)

### Machine Learning Modell

Zum Einsatz kommt hier ein bereits trainiertes Machine Learning Modell. Das Modell wurde duch Zhang et al. entwickelt [1]. Benutzung des ML Algorithmus, wie in `code/image_generation/02_recolor/recolor.py` umgesetzt:

```python
from colorization.colorizers import *

# Recolor steps:
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
out_img = postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())
```

Zu beachten ist, dass das Modell nur mittels moderner Bilder trainiert wurde. Im Training wurde ein Bild im CIELAB Farbraum, dessen Buntheitswerte a* und b* entfernt wurden, als Eingabe für das Modell verwendet. Der Output des Modells wurde mit dem originalen Bild, samt Buntheitswerten, verglichen. Basierend auf dem Vergleich erfolgte das Training des Modells, wie in dieser Abbildung gezeigt [1]:

![ML Training](img_ml_training.png)


Mit dieser Methode ist es nicht möglich das Modell mit den historischen Bildern zu trainieren, da ein Vergleich mit existierenden Buntheitswerten nicht möglich ist.

## 2. Vorbereitung der Stimuli

### Auswahl der Bilder
Es wurden historische und moderne Bilder gewählt. Es dabei wurden 30 historische Bilder aus verschiedenen Quellen zusammengetragen. Das Augenmerk lag darauf, möglichst vielfältige Themen abzudecken. So sind z.B. Architektur-, Landschafts- und Portraitaufnahmen vertreten.

Die Modernen Bilder stellen eine kleinere Kontrollgruppe von 15 Bildern da. Mit diesen Bildern wurde der verwendete Machine Learning Algorithmus trainiert [1]. Die Bilder wurden der Tampere Image Database 2013 entnommen [3]. Die Themen der Bilder entsprechen den Themen der historischen Bilder.

### Ablauf der Erstellung

Wir beschreiben die Erstellung der Stimuli und beziehen uns auf die Ordner in `code/image_generation`. Die nachfolgenden Schritte sind auszuführen zur Erstellung der Stimuli. Jede Datei ist nur einmal auszuführen. Alle Bilddateien in den jeweiligen Ordnern werden jeweils abgearbeitet.

* `00_base_images`: Ordner mit den ursprünglichen Bildern. Hier liegen die modernen Bilder noch als farbige Version vor.
* `01_conversion_modern_images_to_to_bw`: Mit Ausführung von `create_bw_colors.py` werden ausschließlich von den modernen Bildern aus `00_base_images/modern` in schwarz-weiße-Bilder umgewandelt. Die Bilder werden hierzu in den CIELAB-Farbraum konvertiert. Die Buntheitswerte der Bildpunkte werden im Anschluss auf 0 gesetzt. Zum Abschluss werden die Bilder im Unterordner `export` gespeichert.
* `02_recolor`: Zur Rekolorierung der historischen und modernen Bilder `recolor.py` ausführen. Der oben vorgestellte Machine Learning Algorithmus wird verwendet. Die resultierende Bilder werden im Unterordner `export` gespeichert.
* `03_modify_chroma`: Zur Generierung der unterschiedlich bunten Versionen eines Bildes aus dem vorhergenden Schritt `main.py` ausführen. Die resultierenden Bilder sowie eine individuelle Übersicht für jedes Bild werden im Ordner `04_completed_images` gespeichert. Eine Beispielübersicht mit den angewandten Faktoren für die Anpassung der Buntheit:

![Beispiel Übersicht](img_overview.png)


## 3. Experimentelles Design

Unseren Versuch haben wir in zwei verschiedene Experimente aufgeteilt. Einmal das Hauptexperiment „Neuner“ und das Kontrollexperiment „Single“.
Die Testbilder werden mit der Python Anwendung `code/nines/generate_pictures_file.py` in die richtige Form und Größe gebracht, damit sie später bei den Versuchen benutzt werden können. Außerdem wird eine CVS Datei kreiert, welche der Neuner zur Nutzung der Bilder benötigt.

### Neuner
In unserem Hauptexperiment sehen Versuchspersonen jeweils dasselbe Bild in neun verschiedenen Buntheiten. Das originale Schwarz-Weiß-Bild ist in diesen Bildergruppen nicht vorhanden. Die Versuchsperson soll bei jeder Bildgruppe entscheiden welches Bild am realistischsten empfunden wird und mit einem Klick auf eine Zahl von eins bis neun die Entscheidung bestätigen.

![Beispiel einer Bildergruppe](niner_beispiel.PNG)

Es wurden insgesamt 45 Bilder gezeigt. Aufgeteilt wurden diese Bilder in 30 historische und 15 moderne. Die Bilder wurden in jedem Durchlauf zufällig angeordnet, so dass die Chance minimiert wird, das Versuchspersonen beeinflusst werden durch vorherige Durchläufe.

Um selbst einen Versuch durchzuführen, müssen die Bilder im Pictures Ordner inklusive der `pictures.csv` gespeichert werden. Die Anwendung `code/nines/experiment_nines.py` kann dann gestartet werden. Diese liest `pictures.csv` aus und zeigt dem User die Bilder. Außerdem werden alle Eingaben des Users gespeichert und am Ende in einer CSV Datei im results Order gespeichert. Bei einem Neustart des Experiments werden die Bildergruppen wieder zufällig aneinandergereiht.

### Single
Bei unserem Kontrollexperiment wurde überprüft wie Bunt die Versuchspersonen das vom Algorithmus eingefärbte Bild empfinden. Hier haben wir für jede Bildergruppe lediglich ein Bild gezeigt und die Testperson sollte auf einer Skala von 0 (keine Farben) bis 9 (höchste Intensität an Buntheit) ihre Bewertung speichern.

![Beispiel eines Bildes](single_beispiel.PNG)

Die Daten, welche wir durch den Single-Versuch bekommen haben, konnten wir mit denen des Neuners kombinieren, um interessante Schlüsse für unsere Hypothese zu ziehen.

Zum Starten des Kontrollexperiments müssen die Bilder, welche bewerten werden sollen, in den data Ordner. Die Anwendung `code/single_assessment/rating_experiment_single.py` zeigt der Testpersonen die Bilder in einer zufälligen Reihenfolge und speichert die Ergebnisse im result Order als CSV Datei.


## 4. Ergebnisse

Nachdem eine befragte Person an dem Neuener Experiment teilgenommen hat, könnte das Ergebnis für die historischen Bilder exemplarisch folgendermaßen aussehen:

![Neuner - Ergebnisse einer Versuchsperson](niner_assessment_single_observer.png)

Auf der x-Achse sind hier die Bild-IDs abgebildet und auf der y-Achse der normierte gewählte Chromafaktor.
Bei Chromafaktor 0.0 befindet sich eine blaue Linie. Diese symbolisiert den Chromawert, mit dem das Bild vom Algorithmus eingefärbt wurde.
Die Punkte in dem Diagramm zeigen dann für die Antworten der Testperson.

Von uns wurden so 23 Versuchspersonen befragt und die Ergebnisse sind im folgenden Diagramm zusammengetragen.

![Neuner - Historische Bilder - Versuchsergebnisse](niner_assessment_historic_images.png)

Es wurde schnell deutlich, dass für jedes Bild der Median der gewünschten Chromawerte über dem vom Algorithmus generierten Chromawert liegt.
Das bedeutet, dass sich die Versuchspersonen im Median eine buntere Version des Bildes als realistischster empfinden, als das Bild, das von Algorithmus generiert wurde.

Die zwei Bilder, die am weitesten außen des Antwortspektrums liegen, sind folgende:

![H25 - 1.0](H25_1.0_chroma.JPG)

H25 wurde als das unrealistische Bild wahrgenommen. Hier halten die Versuchspersonen im Median folgende Version als am realistischsten:

![H25 - 2.13](H25_2.13_chroma.JPG)

H11 dagegen wurde als das Bild bewertet, bei dem der Algorithmus die realistischste Version generiert hat.

![H11 - 1.0](H11_1.0_chroma.JPG)

Für die modernen Bilder konnten wir ein ganz ähnliches Ergebnis erzielen.

Auch hier wählten die Versuchspersonen im Median meistens buntere Versionen, als die von Algorithmus generiert.

Zu den interessantesten Sonderfällen zählen hier I08, welches am unrealistischsten wahrgenommen wurde.
Hier haben die Probanden eher diese Version als realistisch empfunden.

Das realistischste Bild ist I06 bei dem der Median genau auf dem Bild liegt, welches vom Algorithmus erzeugt wurde.

Als Ausreißer kann man an dieser Stelle noch I23 ansprechen. Bei diesem Bild habe die Versuchspersonen eine unbuntere Version als realistischer empfunden, als das vom Algorithmus generierte.

Im folgenden Diagramm sieht man die vereinigten Ergebnisse für die modernen und die historischen Bilder nebeneinander.
Hier kann man deutlich erkennen, dass Versionen, die vom Algorithmus erzeugt wurden, generell zu unbunt sind, um für die Testpersonen als die realistischste wahrgenommen zu werden.

Ein t-Test zeigt hierbei einen signifikanten Unterschied zwischen den beiden Bildgruppen.

Man sieht, dass die modernen Bilder generell als bunter wahrgenommen werden als die historischen.
Auch hier zeigt ein t-Test einen signifikanten Unterschied.



## 5. Diskussion
Im Rahmen unseres Seminarprojekts haben die Auswertungen unserer Experimente gezeigt, dass Buntheit einen positiven Einfluss auf den wahrgenommenen Realismus hat. Anfänglich sind wir davon ausgegangen, dass wenn ein nachträglich eingefärbtes Bild bunter ist, es als realistischer wahrgenommen wird. Diese Erwartung hat sich nach den gewonnenen Erkenntnissen als wahr herausgestellt, jedoch nicht uneingeschränkt. Wir können zwar die Buntheit eines Bildes beliebig viel hochdrehen, aber ab einem bestimmten Punkt wird das Bild extrem unrealistisch, da die Farben ziemlich grell werden.

Wir stellen fest, dass wir die rekolorierten Bilder aus dem Machine Learning Algorithmus [1] bunter gestalten können und sie dadurch realistischer wirken. Folglich können wir sagen, dass wir mit unserer Vorgehensweise den Algorithmus von Zhang et al., bezogen auf den wahrgenommenen Realismus, verbessern können.



### Mögliche Probleme 
Es war nicht leicht die Aufgabenstellung unserer beiden Experimente kurz und genau zu erklären.
Dies führte dazu, dass im Nachhinein manchmal eine erneute mündliche Erklärung erfolgen musste, um genau klarzustellen, wonach im Experiment gefragt wird.
In einem Fall mussten wir die Ergebnisse des "Single" Kontrollexperiments verwerfen, da die Aufgabenstellung falsch interpretiert worden war.
Damit solche Missverständnisse vermieden werden können, wäre es sinnvoll eine kurze Erläuterung anhand eines Beispiels zum Experiment hinzuzufügen.

In unseren Bildergruppen hatten wir außerdem bekannte Persönlichkeiten, wie zum Beispiel Elvis. Es könnte sein,
dass die Versuchsperson Elvis vorher nur in Schwarz-Weiß kannte, welches dazu führen könnte, dass sie ein etwas unbunteres Bild
als realistischer empfindet. 

### Offene Fragen
In weiterführenden Untersuchungen kann betrachtet werden, ob die Buntheitsanpassung dynamisch für einen Bildpunkt erfolgen kann. So könnte die Nachbarschaft eines Bildpunktes berücksichtigt werden und Bildpunkte in weniger bunten Bereiches eines Bild bunter gemacht werden, als andere Bereiche.

## Referenzen
[1] Zhang et al. Colorful Image Colorization, ECCV Proceedings, 2016, [doi](https://doi.org/10.1007/978-3-319-46487-9_40).

[2] scikit-image development team. Besucht am 21.03.2022, https://scikit-image.org/.

[3] Tampere Image Database. Besucht am 13.12.2021, https://www.ponomarenko.info/tid2013.htm.

[4] Konica Minolta. Precise Color Communication. Besucht am 08.12.2021, https://www.konicaminolta.com/instruments/download/booklet/index.html.

[5] Eva Lübbe. Farbempfindung, Farbbeschreibung und Farbmessung. 1. Auflage, Wiesbaden 2013.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import random

import pyglet
from pyglet import window
from pyglet.window import key

import sys
import time

instructions = """
Sie werden immer neun Fotos des selben Motivs sehen. \n
Auf welchem Foto empfinden Sie die Farben am realistischsten, also so, als würden Sie das Motiv in der Wirklichkeit selber sehen.\n
Drücken Sie 1 - 9 auf Ihrer Tastatur um Ihre Auswahl zu speichern.\n
\n
Drücken Sie ENTER zum Starten \n
Drücken Sie ESCAPE zum Abbrechen"""

end = """ Vielen Dank für Ihre Teilnahme! """

debug = False

class Experiment(window.Window):

    pictures = []
    pictures_buntheit = []
    pictures_group = 0

    def __init__(self, *args, **kwargs):
        self.win = window.Window.__init__(self, *args, **kwargs)

        self.welcome_text = pyglet.text.Label(instructions,
                                              font_name="Arial", multiline=True,
                                              font_size=25, x=int(self.width / 2.0), y=int(self.height / 2.0),
                                              width=int(self.width * 0.75), color=(0, 0, 0, 255),
                                              anchor_x="center", anchor_y="center")

        self.end_text = pyglet.text.Label(end,
                                              font_name="Arial", multiline=True,
                                              font_size=25, x=int(self.width / 2.0), y=int(self.height / 2.0),
                                              width=int(self.width * 0.75), color=(0, 0, 0, 255),
                                              anchor_x="center", anchor_y="center")
        # experiment control
        self.experimentPhase = 0
        self.firstFrame = True

        # Results file - assigning filename
        if debug:
            filename = "results/DEBUGRESULT.csv"
        else:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = "results/result_" + timestr + ".csv"
            
        # Csv writer für unsere Resultate
        self.rf = open(filename, "w", newline="")
        self.resultwriter = csv.writer(self.rf)
        header = ["Bildergruppe", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Auswahl"]
        self.resultwriter.writerow(header)

        # liest Csv mit Bildern und befüllt die Arrays
        self.loaddesign()

        self.dispatch_event("on_draw")

    # liest die csv Datei aus
    def loaddesign(self):
        with open("Pictures/pictures.csv", "r") as f:
            reader = csv.reader(f)
            temp = list(reader)
            for i in temp:
                bunttemp = []
                picturetemp = []
                for pic in i:
                    img = pic.strip("[, ], ").split(",")
                    bunttemp.append(int(img[0]))
                    picturetemp.append(img[1].strip("', "))
                self.pictures_buntheit.append(bunttemp)
                self.pictures.append(picturetemp)
            kombi = list(zip(self.pictures, self.pictures_buntheit))
            random.shuffle(kombi)
            self.pictures, self.pictures_buntheit = zip(*kombi)


    def loadImages(self):
        """ Loads images of current trial """
        # load files
        if debug:
            print('loading files')
        self.images = []
        for i in range(9):
            #ladet das korrekte Bild und speichert die x und y koordinaten
            try:
                temp = pyglet.image.load(self.pictures[self.pictures_group][i])
                self.images.append(temp)
            except IndexError:
                pyglet.app.exit()

    def update(self, dt):
        pass

    def on_draw(self):

        pyglet.gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.clear()

        # Willkommensscreen
        if self.experimentPhase == 0:
            if debug:
                print('experiment phase 0: welcome')
            self.welcome_text.draw()

        # User wählt Bilder aus mit numpad bzw. Zahlen 1-9
        elif self.experimentPhase == 1:
            if debug:
                print('experiment phase 1: going through the trials')
            # platziert die Bilder an den Positionen 1 bis 9
            x = [[0.2, 0.825], [0.5, 0.825], [0.8, 0.825],
                 [0.2, 0.5], [0.5, 0.5], [0.8, 0.5],
                 [0.2, 0.175], [0.5, 0.175], [0.8, 0.175]]
            if self.firstFrame:
                # ladet die bilder in self.images
                self.loadImages()
                self.showNumbers(arr=x)
            for i, image in enumerate(self.images):
                image.blit(int(self.width * x[i][0]) - (image.width // 2), int(self.height * x[i][1]) - (image.height // 2))
            self.firstFrame = False

        # Ende des Versuchs
        elif self.experimentPhase == 2:
            if debug:
                print('experiment phase 2: goodbye')
            sys.exit("Experiment erfolgreich durchgeführt!")

        self.flip()

    def on_key_press(self, symbol, modifiers):
        """ Executed when a /key is pressed"""
        # Programm abbrechen
        if symbol == key.ESCAPE:
            self.dispatch_event('on_close')
        # Enter um vom Willkommen Screen weiter zu kommen
        elif symbol == key.ENTER and self.experimentPhase == 0:
            if debug:
                print("ENTER")
            self.experimentPhase += 1
        # User wählt 1 aus
        elif (symbol == 49 or symbol == 65457) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][0])
        # User wählt 2 aus
        elif (symbol == 50 or symbol == 65458) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][1])
        # User wählt 3 aus
        elif (symbol == 51 or symbol == 65459) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][2])
        # User wählt 4 aus
        elif (symbol == 52 or symbol == 65460) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][3])
        # User wählt 5 aus
        elif (symbol == 53 or symbol == 65461) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][4])
        # User wählt 6 aus
        elif (symbol == 54 or symbol == 65462) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][5])
        # User wählt 7 aus
        elif (symbol == 55 or symbol == 65463) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][6])
        # User wählt 8 aus
        elif (symbol == 56 or symbol == 65464) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][7])
        # User wählt 9 aus
        elif (symbol == 57 or symbol == 65465) and self.experimentPhase == 1:
            self.firstFrame = True
            self.savetrial(resp=self.pictures_buntheit[self.pictures_group][8])
        # wird aufgerufen damit sich das Programm nach dem Keystroke updated

        self.dispatch_event("on_draw")

    # Kreiert Labels und packt sie unter die Bilder damit die Auswahl angenehmer ist fürs Auge.
    def showNumbers(self, arr):
        for i in range(0, 9):
            self.number = pyglet.text.Label("--- " + str(i + 1) + " ---",
                                              font_name="Arial", multiline=True,
                                              font_size=15, x=int(self.width * arr[i][0]) + 700, y=int(self.height * arr[i][1]) - 170,
                                              width=int(self.width * 0.75), color=(0, 0, 0, 255),
                                              anchor_x="center", anchor_y="center")
            self.number.draw()

    # speichert das Ergebnis für die jeweiligen Bilder
    def savetrial(self, resp):
        print("User hat bei Bildergruppe", self.pictures_group, resp, "als bestes befunden")
        result = [self.pictures[self.pictures_group][0].split("/")[1]]
        werte = []
        for i in self.pictures[self.pictures_group]:
            x = i.strip(".jpg").split("/")
            werte.append(x[2])
        wahl = werte[resp]
        werte.sort()
        result.extend(werte)
        result.append(wahl)
        self.resultwriter.writerow(result)
        self.pictures_group += 1


if __name__ == '__main__':

    height = 900 if debug else 1080
    width = 1800 if debug else 1920
    fullscreen = not debug

    win = Experiment(caption="Testumgebung",
                     vsync=False, height=height, width=width, fullscreen=fullscreen)

    pyglet.app.run()

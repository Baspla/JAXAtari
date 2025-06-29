# Links
## Wikipedia
https://de.wikipedia.org/wiki/Draughts
## Spielbare Version
https://atarionline.org/atari-2600/video-checkers
## Manual
https://www.atariage.com/manual_html_page.php?SoftwareID=1427
## ALE Documentation
https://ale.farama.org/environments/video_checkers/

# Sprites
- Spritesheet Spielfeldobjekte
  - 0 = leeres Feld
  - 1 = Spielsteine Weiß
  - 2 = Spielsteine Schwarz
  - 3 = König Weiß
  - 4 = König Schwarz
  - 5 = Cursor Weiß
  - 6 = Cursor Schwarz
- Spielfeld
- Spritesheet Characters
  - Zahlen für Start und End-Position des Zuges
  - J & P für Schlagzwang

## Farben
- Hintergrund: AC5030

# Wie das Spiel funktioniert
Wie beim klassischen Damespiel handelt es sich bei Draughts um ein Brettspiel
für zwei Spieler, die sich am Spielbrett gegenübersitzen.
Dabei spielt ein Spieler die weißen und der andere die schwarzen Spielsteine
und die beiden Spieler machen abwechselnd jeweils einen Zug.
In der Startaufstellung werden die jeweils 12 Spielsteine auf den schwarzen
Feldern der jeweils ersten bis dritten Reihe beider Spielbrettseiten aufgebaut.

Die Farben werden ausgelost oder gewählt,
der schwarze Spieler beginnt das Spiel.
Beide Spieler ziehen nun abwechselnd jeweils einen Stein, wobei die Steine
diagonal vorwärts auf den schwarzen Feldern bewegt werden dürfen.
Wenn ein Spieler die Grundlinie der gegenüberliegenden Seite erreicht,
wird sein Stein zu seinem „king“ (=„König“).
Dieser darf, anders als beim klassischen Damespiel,
auch jeweils nur ein Feld jede diagonale Richtung ziehen,
allerdings anders als die normalen Steine auch rückwärts.

Wie bei der klassischen Dame muss ein Spielstein geschlagen werden,
wenn ein Stein über ihn auf ein dahinter liegendes freies Feld springen kann.
Dabei herrscht Schlagzwang, ein Stein muss also geschlagen werden,
wenn dies möglich ist. Sind nach dem Schlagen weitere Schlagzüge möglich, 
müssen diese auch ausgeführt werden.
Der König kann anders als beim klassischen Damespiel ebenfalls nur über
ein Feld schlagen, dabei jedoch vorwärts wie rückwärts ziehen.
Hat ein Spieler mehrere Schlagmöglichkeiten, darf er wählen, welche er nutzt,
unabhängig von der möglichen Anzahl weiterer Schlagzüge.
Alle geschlagenen Steine werden nach dem Zug vom Spielfeld entfernt.

Wie beim klassischen Damespiel gewinnt der Spieler, dem es gelingt,
möglichst alle Steine des Gegners zu schlagen oder unbeweglich zu machen.
Zudem ist es möglich, das Spiel zu gewinnen, wenn der Gegner nur noch 
einen einzigen verbliebenen Stein hat.


# Eigenheiten
- Bei Schwarz/Weiß Modus wird von oben links nach unten rechts gezählt.
- Beim Farben-Modus wird von unten rechts nach oben links gezählt.
- Es ist NICHT Dame sondern Draughts!!!
- Schwarz beginnt das Spiel.
- Es gibt Schlagzwang
- Alle Steine vorwärts ziehen, Könige vorwärts und rückwärts.
- Bild wird schwarz wenn der CPU am Zug ist. Das übernehmen wir hier nicht.
- Ziffern oben links und

## Zum Game Select Switch (CPU Verhalten)

Use the game select switch to change the game and the game number
(displayed at the upper left corner of the screen as shown in the
diagram).  If the game number is white, then the human player (or the
left player) controls the white pieces on the board.  If the game
number is blue (or black on black-and-white television sets) then the
human is red (grey).  The different game variations are explained in
the GAME VARIATIONS section of this instruction booklet.

[Image of the board.  The game number is in the upper left corner.  The
no. of players is in the upper right corner.  The cursor is the "X" on
the screen.]

In Games 1-9, the computer plays regular checkers.  The computer's
skill level increases as the game number increases.  Game 10 is for two
players.  The number of players for each game is displayed at the upper
right corner of the screen.  (See the diagram on the previous page.)

Games 11-19 are losing or "giveaway" checkers.  As in Games 1-9, the
skill level increases as the game number increases.  The object of
giveaway checkers is to be the first player to be unable to move by
losing all of your pieces or by being blocked.

The game select switch may be used in the middle of a game and the
computer will continue to play using the new game difficulty level or
variation.  When the computer is computing its next move, the game
select switch has no effect.

# Animationsverhalten
Bei einem Gegnerzug wird zwischen 2 Animationszuständen gewechselt:
- Startposition (weißes X) und die verlorenen Stein
- Spielercursor (rotes X) und die Endposition des Zuges

Ist ein Stein nicht ausgewählt, wechselt die Animation zwischen Stein und Cursor.
Ist ein Stein ausgewählt und nicht bewegt, Blinkt der Stein zwei mal kurz und ist dann lang sichtbar.
Ist ein Stein ausgewählt und bewegt, wechselt die Animation zwischen Stein am Ziel mit Cursor am Start und Stein am Start ohne Cursor.

# Architektur
## Internal State
- Array mit 32 Elementen für das Spielfeld. (Nur schwarze Felder)
  - 0 = leer
  - 1 = weißer Stein
  - 2 = weißer König
  - 3 = schwarzer Stein
  - 4 = schwarzer König
- Position des Cursors
- Ausgewählter Stein (0-31)
- Animation Frame
- Gegnerischer Zug Infos
  - Start Position
  - End Position
  - Steintyp am Ende des Zuges (König oder normal)
  - Array an Positionen der geschlagenen Steine
  - Wird der Gegnerzug angezeigt (Er wird angezeigt bis der Spieler eine Aktion ausführt)
- Wer hat gewonnen (0 = keiner, 1 = Weiß, 2 = Schwarz, 3 = Unentschieden, gibt es das?)
## Observation State
- Array mit 32 Elementen für das Spielfeld. (Nur schwarze Felder)
  - 0 = leer
  - 1 = weißer Stein
  - 2 = weißer König
  - 3 = schwarzer Stein
  - 4 = schwarzer König
  - 5 = Schwarzer Cursor
  - 6 = Weißer Cursor
- Zahl für Startposition des Zuges
- Zahl für Endposition des Zuges
- Schlagzwang (0 = nicht aktiv, 1 = aktiv)
## CPU Verhalten
### Wie findet die CPU den "besten" Zug?
Im Original hat er je nach Spielstärke eine bestimmte Anzahl an Zügen die er vorrausberechnet.
### Tims Ansatz
Zuerst eine Funktion die alle möglichen Züge findet. (Die sind in der Regel nicht so viele, vor allem mit Schlagzwang)
Dann eine Score Funktion die für jeden Zug die Bewertung berechnet.
Dann den Zug mit der höchsten Bewertung auswählen.
In der Score funktion wirds dann so komplex wie wirs es uns zutrauen.
Also Verhalten wie "Möglichst viele Steine schlagen" oder "nicht geschlagen werden" oder "König werden" oder "Gegner blockieren" oder "eigene Steine schützen".
Schwerer sind Verhalten die den Gegner betreffen wie "Gegner blockieren" oder "Gegner König werden lassen".
Das könnte dann noch über mehrere Züge hinweg bewertet werden, da gehts dann aber schon in Richtung Minimax Algorithmus.

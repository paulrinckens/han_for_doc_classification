from han.HAN import HAN

han = HAN()

han.load_model(model_dir="./models",
               model_filename="han-10kGNAD.h5",
               tokenizer_filename="tokenizer.pickle")

class_names = ['Etat', 'Inland', 'International', 'Kultur', 'Panorama', 'Sport',
               'Web', 'Wirtschaft', 'Wissenschaft']

probs = han.predict([
                        "Vorwurf lautet auf Untreue – Ragger soll für Frau wegen Rückzahlung eines Wohnbaudarlehens interveniert haben. Klagenfurt – Die Staatsanwaltschaft Klagenfurt hat einen Strafantrag gegen den Kärntner  FPÖ-Chef und Landesrat Christian Ragger wegen Untreue eingebracht. Wie der ORF  Kärnten am Mittwoch meldete, soll Ragger bei einem Beamten interveniert haben.  Es ging um die Rückzahlung eines Wohnbaudarlehens. Ragger wies den Vorwurf  gegenüber der APA zurück. Ragger soll konkret dafür interveniert haben, dass eine Darlehensbezieherin  14.000 Euro für eine vorzeitige Rückzahlung nicht begleichen muss, so der  Bericht des ORF. Vonseiten des Landesgerichts Klagenfurt war für die APA vorerst  niemand erreichbar. Ragger betonte, dass kein Euro Schaden für das Land entstanden sei. Er habe  der behinderten Frau helfen wollen, außerdem liege der Fall bereits Jahre  zurück. Der Prozess hat 2009 begonnen – die Frau hatte ein Wohnbaudarlehen  laufen, und während sie die Wohnung nicht bewohnt hat, hat sie diese vermietet.  Das war aber rechtlich nicht in Ordnung, weswegen sie das Darlehen zurückzahlen  musste, legte Ragger seine Sicht der Dinge dar. Zu Beginn des Prozesses gegen  die Frau habe er mehrmals versucht, eine Erleichterung bei der Rückzahlung zu  erreichen, es sei vor allem um die Zinsen gegangen. Schließlich wurde aber  entschieden, dass die Zinsen ebenfalls zu zahlen sind – und das ist auch so  geschehen."])[
    0]

print(dict({class_names[i]: probs[i] for i in range(len(probs))}))

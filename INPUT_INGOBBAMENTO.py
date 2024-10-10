import pandas as pd
percorso_cartella = input ("Incollare il percorso della cartella in cui si trova il file di input SEZIONE : ")

percorso_file = percorso_cartella+"/SEZIONE.CSV"

data = pd.read_csv(percorso_file,sep = ';')   


INPUT = {
	"NUMERO_DI_PUNTI" : eval(data.VALORE[0].strip()),	#[numero puro]
#
	"X_PUNTI": eval(data.VALORE[1].strip()), #[cm]
#
	"Y_PUNTI" : eval(data.VALORE[2].strip()), #[cm]
#
	"NUMERO_DI_ASTE" : (data.VALORE[3].strip()), #[numero puro]
#
	"CONNETTIVITA" : eval(data.VALORE[4].strip()),	#
#
	"SPESSORI" : eval(data.VALORE[5].strip()) , #[cm]
#
    "SCALA_DIAGRAMMA" :			eval(data.VALORE[6].strip())					,#	[numero puro]
#
}
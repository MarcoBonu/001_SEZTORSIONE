import pandas as pd
percorso_cartella = input ("Incollare il percorso della cartella in cui si trova il file di input SCHEMA STATICO : ")

percorso_file = percorso_cartella+"/SCHEMA_STATICO.CSV"

data = pd.read_csv(percorso_file,sep = ';')   


INPUT = {
	"LUNGHEZZA_TRAVE" : data.VALORE[0].strip(),	#[m]
 #
	"ESTREMO_DESTRO_VINCOLATO": data.VALORE[1].strip(), #[S ,N ]
#
	"NUMERO_MOMENTI_TORCENTI_CONCENTRATI" : data.VALORE[2].strip(), #[numero puro]
#
	"INTENSITA_MOMENTI_TORCENTI_CONCENTRATI" : eval(data.VALORE[3].strip()), #[kNm]
#
	"POSIZIONE_MOMENTI_TORCENTI_CONCENTRATI" : eval(data.VALORE[4].strip()),	#[m]
#
	"MATERIALE" : data.VALORE[5].strip() , #( 1 - Acciaio, 2 - Calcestruzzo )
#
	"TRAVE_PRISMATICA" : data.VALORE[6].strip(), #[S,N]
#
#se la trave è prismatica attivare questa porzione
#
	"JT" : eval(data.VALORE[7].strip()),  #[cm^4]
#
	"Jpsi" : eval(data.VALORE[8].strip()), #[cm^6]
#
#se la trave non è prismatica attivare questa porzione
#
#	"JT" : [99.33 , 101.10,1], #[cm^4]
#
#	"Jpsi" : [2850000 , 408000000,1] #[cm^6]
#
}
##CALCOLO FUNZIONE DI INGOBBAMENTO SEZIONI SOTTILI APERTE
from CLASSI_INGOBBAMENTO import CONTENITORI_INGOBBAMENTO,data_linewidth_plot
from INPUT_INGOBBAMENTO import INPUT,percorso_cartella
import math
import matplotlib.pyplot as plt
from collections import Counter

def Lunghezze(X,Y,C,NUM_ASTE,SPESSORE):
    L=[]

    for i in range(NUM_ASTE):
        I=C[i][0]-1
        J=C[i][1]-1
        
        L.append(math.sqrt((X[J]-X[I])**2+(Y[J]-Y[I])**2))
        
    
    return L

def Area(L,SPESSORE,NUM_ASTE):
    A=[]
    for i in range(NUM_ASTE):
        A.append(L[i]*SPESSORE[i])
    Atot=sum(A)
    return A, Atot

def G(X,Y,C,A,Atot,NUM_ASTE):

    AX=[]
    AY=[]
    xg=[]
    yg=[]
    
    for i in range(NUM_ASTE):
        I=C[i][0]-1
        J=C[i][1]-1

        xg.append((X[J]+X[I])/2)
        yg.append((Y[J]+Y[I])/2)

    for i in range(NUM_ASTE):
        AX.append(A[i]*xg[i])
        AY.append(A[i]*yg[i])

    xG = sum(AX)/Atot
    yG = sum(AY)/Atot
    return [xG, yG, xg , yg]

def rot_J(Jn,Jm,Jnm,alfa):
    Jcsi = Jn*math.cos(-alfa)**2 + Jm*math.sin(-alfa)**2-Jnm*math.sin(-2*alfa)
    Jpsi = Jn*math.sin(-alfa)**2 + Jm*math.cos(-alfa)**2+Jnm*math.sin(-2*alfa)
    Jcsipsi = (Jn-Jm) * math.sin(-alfa) * math.cos(-alfa)+Jnm*(math.cos(-alfa)**2 - math.sin(-alfa)**2)
    return [Jcsi, Jpsi, Jcsipsi]

def J(Lunghezze,SPESSORE, X,Y,C,xG,yG,xg,yg,Aree,NUM_ASTE):

    alfa=[]
    for i in range(NUM_ASTE):
        I=C[i][0]-1
        J=C[i][1]-1
        dY=(Y[J]-Y[I])
        dX=(X[J]-X[I])

        if dX==0:
            alfai=math.pi/2
        else:
            if dY>=0 and dX>=0:
                alfai=math.atan(dY/dX)
            elif dY>=0 and dX<=0:
                alfai=math.atan(dY/dX)+math.pi
            elif dY<=0 and dX<=0:
                alfai=math.atan(dY/dX)+math.pi
            elif dY<=0 and dX>=0:
                alfai=math.atan(dY/dX)+2*math.pi
        alfa.append(alfai)
    
    Jn=[]
    Jm=[]
    Jnm=[]
    for i in range(NUM_ASTE):
        ti=SPESSORE[i]
        Li=Lunghezze[i]

        Jni=(ti**3*Li)/12
        Jmi=(ti*Li**3)/12
        Jnmi=0

        Jn.append(Jni)
        Jm.append(Jmi)
        Jnm.append(Jnmi)

    Jcsi = []
    Jpsi = []
    Jcsipsi = []
    for i in range(NUM_ASTE):
        Ji =rot_J(Jn[i],Jm[i],Jnm[i],alfa[i])
        Jcsi.append(Ji[0])
        Jpsi.append(Ji[1])
        Jcsipsi.append(Ji[2])

    Jx=[]
    Jy=[]
    Jxy=[]
    for i in range(NUM_ASTE):
        Jx.append(Jcsi[i]+Aree[i]*(yG-yg[i])**2)
        Jy.append(Jpsi[i]+Aree[i]*(xG-xg[i])**2)
        Jxy.append(Jcsipsi[i]+Aree[i]*(xG-xg[i])*(yG-yg[i]))
    
    JXtot=sum(Jx)
    JYtot=sum(Jy)
    JXYtot=sum(Jxy)

    dY = -2*JXYtot
    dX = JXtot-JYtot
    if dX==0:
        alfa_princ=math.pi/4
    else:
        if dY>=0 and dX>=0:
            alfa_princ=math.atan((dY)/(dX))/2
        elif dY>=0 and dX<=0:
            alfa_princ=math.atan((dY)/(dX))/2+math.pi
        elif dY<=0 and dX<=0:
            alfa_princ=math.atan((dY)/(dX))/2+math.pi
        elif dY<=0 and dX>=0:
            alfa_princ=math.atan((dY)/(dX))/2+2*math.pi

       

    JPRINC = rot_J(JXtot,JYtot,JXYtot,-alfa_princ)
    J1=JPRINC[0]
    J2=JPRINC[1]
    J12=JPRINC[2]
    return [alfa,Jn,Jm,Jnm,Jcsi,Jpsi,Jcsipsi,Jx,Jy,Jxy,JXtot,JYtot,JXYtot,J1,J2,J12,alfa_princ]

def Spsi(C,NUM_ASTE,PSI,SPESSORE, Lunghezze):
    S_PSI_ASTE=[]
    for i in range(NUM_ASTE):
        I=C[i][0]-1
        J=C[i][1]-1

        ti = SPESSORE[i]
        Li=Lunghezze[i]

        S_PSI_i = ti*Li*(PSI[J]+ PSI[I])/2

        S_PSI_ASTE.append(S_PSI_i)

    S_PSI=sum(S_PSI_ASTE)
    return S_PSI

def PSI(X,Y,C,xG,yG,NUMERO_DI_PUNTI,NUM_ASTE,psi_0,CT_x_rot,CT_y_rot):
    PSI=[]
    for i in range(NUMERO_DI_PUNTI):
        PSI.append(0)

    for i in range(NUM_ASTE):

        I=C[i][0]-1
        J=C[i][1]-1

        if i ==0:
            PSI[I]=psi_0


        XJI=X[J]-X[I]
        YJI=Y[J]-Y[I]

        XCTJ=CT_x_rot-X[J]
        YCTJ=CT_y_rot-Y[J]

        A=((XJI*YCTJ)-(YJI*XCTJ))/2

        PSI[J] = PSI[I]-2*A


    X0G=X[C[0][0]-1]-xG
    Y0G=Y[C[0][0]-1]-yG

    XCT0=CT_x_rot-X[C[0][0]-1]
    YCT0=CT_y_rot-Y[C[0][0]-1]

    AGCT=((X0G*YCT0)-(Y0G*XCT0))/2

    for i in range(len(PSI)):
        PSI[i] = PSI[i]-2*AGCT
    return PSI
       
def grafici(C,X,Y,xG,yG,xg,yg,NUM_ASTE,SPESSORE,alfa_princ,J1,J2,CT_x,CT_y,percorso_cartella):
    spess=max(SPESSORE)
    for i in range(NUM_ASTE):
        I=C[i][0]-1
        J=C[i][1]-1

        data_linewidth_plot([X[I],X[J]],[Y[I],Y[J]], linewidth=SPESSORE[i],color="k",alpha=0.4)

    l_ax=max(X)/math.sqrt(2) 
    ax_princ_1=[xG-l_ax*math.cos(alfa_princ),xG,xG+l_ax*math.cos(alfa_princ)]
    ay_princ_1=[yG-l_ax*math.sin(alfa_princ),yG,yG+l_ax*math.sin(alfa_princ)]
    plt.text(xG-l_ax*math.cos(alfa_princ),yG-l_ax*math.sin(alfa_princ),"ax1")
    plt.plot(ax_princ_1,ay_princ_1,'--',color=[0.5,0.5,0.5])

    ax_princ_2=[xG-l_ax*math.cos(alfa_princ+math.pi/2),xG,xG+l_ax*math.cos(alfa_princ+math.pi/2)]
    ay_princ_2=[yG-l_ax*math.sin(alfa_princ+math.pi/2),yG,yG+l_ax*math.sin(alfa_princ+math.pi/2)]
    plt.text(xG+l_ax*math.cos(alfa_princ+math.pi/2),yG+l_ax*math.sin(alfa_princ+math.pi/2),"ax2")
    plt.plot(ax_princ_2,ay_princ_2,'--',color=[0.5,0.5,0.5])

    plt.text(xG+l_ax*math.cos(alfa_princ),yG+l_ax*math.sin(alfa_princ)-spess*1.0,f"alpha = {int(alfa_princ*180/math.pi)} °")

    plt.text(max(X)+10*spess,min(Y), f"[Gx={round(xG,2)} cm ;Gy={round(yG,2)} cm]",color="r",bbox=dict( facecolor = "w", edgecolor='black'))
    plt.plot(xG,yG,'o',color="r",markersize=10)

    plt.text(max(X)+10*spess,max(Y), f"[CTx={round(CT_x,2)} cm ;CTy={round(CT_y,2)} cm]",color="g",bbox=dict( facecolor = "w", edgecolor='black'))
    plt.plot(CT_x,CT_y,'o',color="g",markersize=10)

    for i in range(len(X)):
        t=plt.text(X[i]+spess,Y[i]+spess,i+1)
        s=plt.text(X[i]+spess,Y[i]+spess,i+1)
        t.set_bbox(dict(alpha=0.4,facecolor="w"))
        s.set_bbox(dict(alpha=0.4,facecolor="w"))

    plt.axis('equal')
    plt.savefig(f"{percorso_cartella}/SEZIONE.png",bbox_inches='tight')

def JT(NUM_ASTE,SPESSORE,Lunghezze):
    JT_asta=[]
    for i in range(NUM_ASTE):
        ti=SPESSORE[i]
        Li=Lunghezze[i]
        JT_asta.append(Li*ti**3/3)

    JTtot = sum(JT_asta)  
    return JTtot  

def grafici_ingobbamento(C,X,Y,xG,yG,xg,yg,NUM_ASTE,SPESSORE,alfa_princ,J1,J2,CT_x,CT_y,PSI,Lunghezze,scala_diagrama,percorso_cartella,Jpsi,Jt):
    plt.clf()
    scala=scala_diagrama
    spess=max(SPESSORE)
    for i in range(NUM_ASTE):
        I=C[i][0]-1
        J=C[i][1]-1

        data_linewidth_plot([X[I],X[J]],[Y[I],Y[J]], linewidth=SPESSORE[i],color="k",alpha=0.4)

    for i in range(NUM_ASTE):
        x_psi=[]
        y_psi=[]
        I=C[i][0]-1
        J=C[i][1]-1

        Li = Lunghezze[i]
        nx =(X[J]-X[I])/(Li)
        ny =(Y[J]-Y[I])/(Li)
        
        tx = - ny
        ty = nx
        x_psi.append(X[I])
        x_psi.append(X[I]+PSI[I]*tx*scala)
        
        y_psi.append(Y[I])
        y_psi.append(Y[I]+PSI[I]*ty*scala)
        
        plt.text(X[I]+PSI[I]*tx*scala,Y[I]+PSI[I]*ty*scala, f"{round(PSI[I],2)} cm^2",color="r" )

        x_psi.append(X[J]+PSI[J]*tx*scala)
        x_psi.append(X[J])
        y_psi.append(Y[J]+PSI[J]*ty*scala)
        y_psi.append(Y[J])
        
        plt.text(X[J]+PSI[J]*tx*scala,Y[J]+PSI[J]*ty*scala, f"{round(PSI[J],2)} cm^2",color="r" )

        plt.plot(x_psi,y_psi,color="r")

    plt.text(max(X)+10*spess,max(Y),f"Jpsi = {int(Jpsi)} cm^6",bbox=dict( facecolor = "w", edgecolor='black'))
    plt.text(max(X)+10*spess,min(Y),f"Jt = {int(Jt)} cm^4",bbox=dict( facecolor="w",edgecolor='black'))

    plt.axis('equal')
    plt.savefig(f"{percorso_cartella}/DIAG_INGOBBAMENTO.png",bbox_inches='tight')

def int_fg(SPESSORE,Lunghezze,FUNC1,FUNC2,NUM_ASTE,C):

    int_fg_list=[]
    
    for i in range(NUM_ASTE):
        I=C[i][0]-1
        J=C[i][1]-1

        ti = SPESSORE[i]
        Li = Lunghezze[i]

        f0 = FUNC1[I] #ad esempio FUNC1  è PSI, cioè la lista di tutti i valori PSI per ogni nodo
        f1 = (FUNC1[J]-FUNC1[I])/Li
        g0 = FUNC2[I] #ad esempio FUNC1  è X2, cioè la lista di tutti le posizioni X2 per ogni nodo
        g1 = (FUNC2[J]-FUNC2[I])/Li

        int_fg_i = ti*(f0*g0*Li+(f0*g1+f1*g0)*(Li**2)/2+f1*g1*(Li**3)/3)
        int_fg_list.append(int_fg_i)

    int_fg = sum(int_fg_list)
    return int_fg

def rototr(X,Y,xG,yG,alfa):
    x1=[]
    x2=[]

    for i in range(len(X)):
        x1.append((X[i]-xG)*math.cos(alfa)+(Y[i]-yG)*math.sin(alfa))
        x2.append(-(X[i]-xG)*math.sin(alfa)+(Y[i]-yG)*math.cos(alfa))

    return [x1,x2]

def main():

    DATI = CONTENITORI_INGOBBAMENTO()

    for k, v in INPUT.items():
        setattr(DATI, k, v)

    #Assegno le variabili
    N=DATI.NUMERO_DI_PUNTI
    X=DATI.X_PUNTI
    Y=DATI.Y_PUNTI
    NUM_ASTE=int(DATI.NUMERO_DI_ASTE)
    C=DATI.CONNETTIVITA
    SPESSORE=DATI.SPESSORI
    scala_diagrama = DATI.SCALA_DIAGRAMMA			
    nodi_J = []
    for i in range(NUM_ASTE):
        nodi_J.append(C[i][1])

    if max(Counter(nodi_J).values())>1:
        I=nodi_J.index(max(Counter(nodi_J).values()))
        print(f"Attenzione, il nodo {nodi_J[I]} è usato per {max(Counter(nodi_J).values())} volte come nodo di arrivo. Ridefinire la connettività attorno al nodo {nodi_J[I]}.")
        quit()

    #Calcolo le proprietà principali
    L=Lunghezze(X,Y,C,NUM_ASTE,SPESSORE)

    A=Area(L,SPESSORE,NUM_ASTE)
    Aree=A[0]
    Atot=A[1]

    B=G(X,Y,C,Aree,Atot,NUM_ASTE)
    xG=B[0]
    yG=B[1]
    xg=B[2]
    yg=B[3]

    Mom_inerzia = J(L,SPESSORE, X,Y,C,xG,yG,xg,yg,Aree,NUM_ASTE)
    J1=Mom_inerzia[-4]
    J2=Mom_inerzia[-3]
    J12=Mom_inerzia[-2]
    alfa_princ= Mom_inerzia[-1]
    
    
    #Calcolo della funzione di ingobbamento corretta
    #Si individuano due valori a caso metendo la costante dapprima pari a 0 e poi pari a 1000
    a=0
    b=1000
    PSI_1 = PSI(X,Y,C,xG,yG,N,NUM_ASTE,a,xG,yG)
    PSI_2 = PSI(X,Y,C,xG,yG,N,NUM_ASTE,b,xG,yG)

    S_PSI_1 = Spsi(C,NUM_ASTE,PSI_1,SPESSORE, L)
    S_PSI_2 = Spsi(C,NUM_ASTE,PSI_2,SPESSORE, L)

    PSI_0 = b-S_PSI_2*((b-a)/(S_PSI_2-S_PSI_1))
    #Funzione di ingobbamento corretta
    PSI_CORRETTO = PSI(X,Y,C,xG,yG,N,NUM_ASTE,PSI_0,xG,yG)
    
    #Devo cambiare sistema di riferimento in modo da esprimere le posizioni dei nodi nel sistema di riferimento principale della trave
    x_princ=rototr(X,Y,xG,yG,alfa_princ)
    x1 = x_princ[0]
    x2 = x_princ[1]

    #Ora posso calcolare i momenti J1psi,J2psi
    J1PSI = int_fg(SPESSORE,L,PSI_CORRETTO,x2,NUM_ASTE,C)
    J2PSI = int_fg(SPESSORE,L,PSI_CORRETTO, [-i for i in x1] ,NUM_ASTE,C)

   

    #Posizione del centro di taglio
    CT_x=-J1PSI/J1
    CT_y=-J2PSI/J2

    
    CT_x_rot =  rototr(rototr([CT_x],[CT_y],0,0,-alfa_princ)[0],rototr([CT_x],[CT_y],0,0,-alfa_princ)[1],-xG,-yG,0)[0]
    CT_y_rot =  rototr(rototr([CT_x],[CT_y],0,0,-alfa_princ)[0],rototr([CT_x],[CT_y],0,0,-alfa_princ)[1],-xG,-yG,0)[1]

    grafici(C,X,Y,xG,yG,xg,yg,NUM_ASTE,SPESSORE,alfa_princ,J1,J2,CT_x_rot[0],CT_y_rot[0],percorso_cartella )

    #Ora è possibile calcolare psi rispetto al centro di taglio
    
    PSI_CENTRO_TAGLIO = PSI(X,Y,C,xG,yG,N,NUM_ASTE,PSI_0,CT_x_rot[0],CT_y_rot[0])
    J_PSI_CT = int_fg(SPESSORE,L,PSI_CENTRO_TAGLIO,PSI_CENTRO_TAGLIO,NUM_ASTE,C)
    S_PSI_CT = Spsi(C,NUM_ASTE,PSI_CENTRO_TAGLIO,SPESSORE, L)
    Jt = JT(NUM_ASTE,SPESSORE,L)
    print(f"Jpsi = {J_PSI_CT} cm^6")
    print(f"Jt = {Jt} cm^4")
    grafici_ingobbamento(C,X,Y,xG,yG,xg,yg,NUM_ASTE,SPESSORE,alfa_princ,J1,J2,CT_x,CT_y,PSI_CENTRO_TAGLIO,L,scala_diagrama,percorso_cartella,J_PSI_CT,Jt)
    return PSI_CENTRO_TAGLIO, C, SPESSORE, J1, J2, alfa_princ,NUM_ASTE
    

import cv2 as cv
import numpy as np
from numpy import asarray
#from PIL import image
import easyocr
from tkinter import *
import os
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
from forex_python.converter import CurrencyRates

gui = Tk(className='Conversie valutara')
reset=1
cercLeu=["Cerc 1leu","Cerc 5lei","Cerc 10lei","Cerc 20lei","Cerc 50lei","Cerc 100lei","Cerc 200lei","Cerc 500lei"]
steagEuro=["Steag 5euro","Steag 10euro","Steag 20euro","Steag 50euro","Steag 100euro","Steag 200euro","Steag 500euro"]
coroaneLire=["Coroana 5lire","Coroana 50lire"]
semneDolari=["Semn 1dolar"]
monede=["BANI","CENT EURO"]

ValoriEuro=["1","2","5","10","20","50","100","200","500"]
ValoriLeu=['1','5',"10","20","50","100","500"]
ValoriLire=["5","10","20","50"]
ValoriDolari=["1","5","10","20","50","100"]
ValoriRuble=["5","10","50","100",'500',"1000","5000"]
ValoriYen=["10","2000","5000","10000"]

Conversii=['EUR','GBP','RON','USD','JPY','RUB']


def ColorToGrayScale(img14):
    grayscale = cv.cvtColor(img14,cv.COLOR_BGR2GRAY)
    return grayscale

def CoinDetection(img15):
    gray = cv.cvtColor(img15, cv.COLOR_BGR2GRAY)  
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    m1=0
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 500, param1=110, param2=35, minRadius=10, maxRadius=600)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv.circle(img15, (x, y), r, (0, 255, 0), 4)
        m1=m1+1
    #cv.imshow('Detectat', img15)
    return m1
    
def Denoising(img16,w,h):
    median = cv.medianBlur(img16,5)
    gauss = cv.GaussianBlur(img16,(5,5),0)
    images=np.concatenate((median,gauss),axis=0)
    return images

def Valuedetection(img32):
   reader=easyocr.Reader(['en'])
   text3 = reader.readtext(img32,paragraph="False")
   return text3

def Binarizare(img17):
    medianValue=128
    (medianValue, binar)=cv.threshold(img17, 127, 255, cv.THRESH_BINARY)
    return binar

def Recognition(img12,img11):
    gray=ColorToGrayScale(img12) 
    w1,h1=img11.shape[::-1] 
    res=cv.matchTemplate(gray,img11,cv.TM_CCOEFF_NORMED)
    th=0.8
    flag = False
    if np.amax(res) > th:
        flag = True
    loc=np.where(res>=th)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img12,pt,(w1 + pt[0],h1+pt[1]),(0,255,255),2)
    return flag

def Recognition1(img12,img11):
    gray=ColorToGrayScale(img12) 
    w1,h1=img11.shape[::-1] 
    res=cv.matchTemplate(gray,img11,cv.TM_CCOEFF)
    th=0.3
    flag = False
    if np.amax(res) > th:
        flag = True
    loc=np.where(res>=th)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img12,pt,(w1 + pt[0],h1+pt[1]),(0,255,255),2)
    return flag

def RegionDivide(img13,w,h):
    blaclPixels=0
    whitePixles=0
    w13=int(w/3)
    w23=2*int(w/3)
    for i in range(0,h-1):
        for j in range(0,w13-1):
            if img13[i][j] == 255:
                whitePixles=whitePixles+1
            else:
                blaclPixels=blaclPixels+1
    rs=float(blaclPixels/whitePixles)
    blaclPixels=0
    whitePixles=0
    for i in range(0,h-1):
        for j in range(w13-1,w23-1):
            if img13[i][j] == 255:
                whitePixles=whitePixles+1
            else:
                blaclPixels=blaclPixels+1
    rc=float(blaclPixels/whitePixles)
    blaclPixels=0
    whitePixles=0   
    for i in range(0,h-1):
        for j in range(w23,w):
            if img13[i][j] == 255:
                whitePixles=whitePixles+1
            else:
                blaclPixels=blaclPixels+1
    rd=float(blaclPixels/whitePixles)
    return rs,rc,rd

def BinarizareInversa(img17):
    #medianValue1=128
    (medianValue, binar1)=cv.threshold(img17, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return binar1

def Open():
    f_types = [('Jpg files','*.jpg')]
    filename=filedialog.askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    img=img.resize((600,300))
    img=ImageTk.PhotoImage(img)
    e1=Label(gui)
    e1.place(x=500,y=30)
    e1.image=img
    e1['image']=img
    if reset==0:
        img.destroy()


def Open1():
    f_types = [('Jpg files','*.jpg')]
    filename=filedialog.askopenfilename(filetypes=f_types)
    img=cv.imread(filename)
    return img

def Recunoaste():
    flag2=0
    img=Open1()
    imggray=ColorToGrayScale(img)
    binar = Binarizare(imggray)
    binar1 = BinarizareInversa(imggray)
    h, w, c = img.shape
    text1=Valuedetection(binar1)
    #n=CoinDetection(img)
    n=3
    if n>1:
        flagd=0
        flags=0
        flagc=0
        if flag2 == 0:
            for i in cercLeu:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == True:
                    flag2=1
                    valoare=""
                    for i1 in range(0,len(text1)):
                        for l in ValoriLeu:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                    text2=valoare+" ron"
                    labe1 = Label(gui, text="Ati introdus "+text2,bg='#0000FF',height=1,widt=30,fg='red',font=32)
                    labe1.place(x=708,y=350)
            
            for i in steagEuro:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=1
                    valoare=''
                    for i1 in range(0,len(text1)):
                        for l in ValoriEuro:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                        if valoare != "":
                            break        
                    text2=valoare+" euro"
                    labe1 = Label(gui, text="Ati introdus "+text2,bg='#0000FF',height=1,widt=30,fg='red',font=32)
                    labe1.place(x=708,y=350)
            
            for i in coroaneLire:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=1
                    valoare=''
                    for i1 in range(0,len(text1)):
                        for l in ValoriLire:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                        if valoare != "":
                            break    
                    text2=valoare+" lire"
                    labe1 = Label(gui, text="Ati introdus "+text2,bg='#0000FF',height=1,widt=22,fg='red',font=32)
                    labe1.place(x=708,y=350)
                    break
            for i in semneDolari:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template1\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=1
                    valoare=''
                    for i1 in range(0,len(text1)):
                        for l in ValoriDolari:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l        
                    text2=valoare+" dolari americani"
                    labe1 = Label(gui, text="Ati introdus "+text2,bg='#0000FF',height=1,widt=30,fg='red',font=32)
                    labe1.place(x=708,y=350)
                    break
        
        if flag2==0:
            rs1,rc1,rd1=RegionDivide(binar,w,h)
            if flagc == 0:
                if(rc1<1 and rd1<1 and rs1<1):
                    flags=1
                    flagc=1
                    flagd=1
                    flags=1
                    flag2=1
                    for i1 in range(0,len(text1)):
                        for l in ValoriRuble:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                            if valoare =='5':
                                valoare='500'       
                    text2=valoare+" ruble rusesti"
                    labe1 = Label(gui, text="Ati introdus "+text2,bg='#0000FF',height=1,widt=30,fg='red',font=32)
                    labe1.place(x=708,y=350)
            if flagd ==0:
                if(rd1<2 and rc1<1 and rs1<1):
                    flags=1
                    flagc=1
                    flagd=1
                    flags=1
                    flag2=1
                    for i1 in range(0,len(text1)):
                        for l in ValoriYen:
                            if text1[i1][1].find(l) >= 0:
                                valoare1=l
                            if valoare1 =='10':
                                valoare1='1000'
                    text2=valoare1+" yeni japonezi"
                    labe1 = Label(gui, text="Ati introdus "+text2,bg='#0000FF',height=1,widt=30,fg='red',font=32)
                    labe1.place(x=708,y=350)

def Conversie():
    flag2=0
    img=Open1()
    imggray=ColorToGrayScale(img)
    binar = Binarizare(imggray)
    binar1 = BinarizareInversa(imggray)
    h, w, c = img.shape
    text1=Valuedetection(binar1)
    #n=CoinDetection(img)
    n=3
    if n>1:
        flagd=0
        flags=0
        flagc=0
        if flag2 == 0:
            for i in cercLeu:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == True:
                    flag2=1
                    valoare=""
                    simbol='RON'
                    y1=150
                    c = CurrencyRates()
                    for i1 in range(0,len(text1)):
                        for l in ValoriLeu:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                    for l in Conversii:
                        if l!=simbol and l!='RUB':
                            convertit=int(valoare)*c.get_rate(simbol,l)
                            text3=str(convertit)+' '+l
                            labe1 = Label(gui, text=text3,bg='#0000FF',height=1,widt=30,fg='red',font=32)
                            labe1.place(x=670,y=290+y1)
                            y1=y1+70
            
            for i in steagEuro:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=1
                    simbol='EUR'
                    valoare=''
                    y1=150
                    c = CurrencyRates()
                    for i1 in range(0,len(text1)):
                        for l in ValoriEuro:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                        if valoare != "":
                            break        
                    for l in Conversii:
                        if l!=simbol and l!='RUB':
                            convertit=int(valoare)*c.get_rate(simbol,l)
                            text3=str(convertit)+' '+l
                            labe1 = Label(gui, text=text3,bg='#0000FF',height=1,widt=27,fg='red',font=32)
                            labe1.place(x=670,y=290+y1)
                            y1=y1+70
            
            for i in coroaneLire:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=1
                    simbol='GBP'
                    valoare=''
                    y1=150
                    c = CurrencyRates()
                    for i1 in range(0,len(text1)):
                        for l in ValoriLire:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                        if valoare != "":
                            break    
                    for l in Conversii:
                        if l!=simbol and l!='RUB':
                            convertit=int(valoare)*c.get_rate(simbol,l)
                            text3=str(convertit)+' '+l
                            labe1 = Label(gui, text=text3,bg='#0000FF',height=1,widt=27,fg='red',font=32)
                            labe1.place(x=670,y=290+y1)
                            y1=y1+70
                    break
            for i in semneDolari:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=0
                    simbol='USD'
                    valoare=''
                    y1=150
                    c = CurrencyRates()
                    for i1 in range(0,len(text1)):
                        for l in ValoriDolari:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l        
                    for l in Conversii:
                        if l!=simbol and l!='RUB':
                            convertit=int(valoare)*c.get_rate(simbol,l)
                            text3=str(convertit)+' '+l
                            labe1 = Label(gui, text=text3,bg='#0000FF',height=1,widt=30,fg='red',font=32)
                            labe1.place(x=670,y=290+y1)
                            y1=y1+70
                    break
        
        if flag2==0:
            rs1,rc1,rd1=RegionDivide(binar,w,h)
            if flagc == 0:
                simbol='JPY'
                valoare=''
                c = CurrencyRates()
                y1=150
                if(rc1<1 and rd1<1 and rs1<1):
                    flags=1
                    flagc=1
                    flagd=1
                    flags=1
                    flag2=1
                    for i1 in range(0,len(text1)):
                        for l in ValoriRuble:
                            if text1[i1][1].find(l) >= 0:
                                valoare=l
                            if valoare =='5':
                                valoare='500'       
                    for l in Conversii:
                        if l!=simbol and l!='RUB':
                            convertit=int(valoare)*c.get_rate(simbol,l)
                            text3=str(convertit)+' '+l
                            labe1 = Label(gui, text=text3,bg='#0000FF',height=1,widt=27,fg='red',font=32)
                            labe1.place(x=670,y=290+y1)
                            y1=y1+70
            if flagd ==0:
                simbol='RUB'
                valoare=''
                c = CurrencyRates()
                y1=150
                if(rd1<2 and rc1<1 and rs1<1):
                    flags=1
                    flagc=1
                    flagd=1
                    flags=1
                    flag2=1
                    for i1 in range(0,len(text1)):
                        for l in ValoriYen:
                            if text1[i1][1].find(l) >= 0:
                                valoare1=l
                            if valoare1 =='10':
                                valoare1='1000'
                    for l in Conversii:
                        if l!=simbol and l!='RUB':
                            convertit=int(valoare1)*c.get_rate(simbol,l)
                            text3=str(convertit)+' '+l
                            labe1 = Label(gui, text=text3,bg='#0000FF',height=1,widt=27,fg='red',font=32)
                            labe1.place(x=670,y=290+y1)
                            y1=y1+70

def Reset():
    gui.delete('all')


if __name__ == "__main__":
    
    gui.geometry("1200x700")
    gui['background']='#0000FF'
    open_file=Button(gui,text='Incarca poza',bg='#5C5C5C',fg='black',height=2,width=45,command=Open)
    open_file.place(x=25,y=75)
    recunoaste=Button(gui,text='Recunoastere',bg='#5C5C5C',fg='black',height=2,width=45,command=Recunoaste)
    recunoaste.place(x=25,y=175)
    converteste=Button(gui,text='Conversie',bg='#5C5C5C',fg='black',height=2,width=45,command=Conversie)
    converteste.place(x=25,y=275)
    gui.mainloop()  
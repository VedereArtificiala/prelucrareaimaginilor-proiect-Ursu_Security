import cv2 as cv
import numpy as np

cercLeu=["Cerc 1leu","Cerc 5lei","Cerc 10lei","Cerc 20lei","Cerc 50lei","Cerc 100lei","Cerc 200lei","Cerc 500lei"]
steagEuro=["Steag 5euro","Steag 10euro","Steag 20euro","Steag 50euro","Steag 100euro","Steag 200euro","Steag 500euro"]
coroaneLire=["Coroana 5lire","Coroana 10lire","Coroana 20lire","Coroana 50lire"]
semneDolari=["Semn 1dolar","Semn 5dolari","Semn 10dolari","Semn 20dolari","Semn 50dolari","Semn 100dolari"]

def ColorToGrayScale(img14):
    grayscale = cv.cvtColor(img14,cv.COLOR_BGR2GRAY)
    return grayscale

def CoinDetection(img15,rmin,rmax):
    gray = cv.cvtColor(img15, cv.COLOR_BGR2GRAY)  
    blurred = cv.medianBlur(gray, 7)  
    return cv.HoughCircles(blurred,  cv.HOUGH_GRADIENT,  1,  40,param1=50,  param2=30, minRadius=61,  maxRadius=91)
    
def Denoising(img16,w,h):
    median = cv.medianBlur(img16,5)
    gauss = cv.GaussianBlur(img16,(5,5),0)
    images=np.concatenate((median,gauss),axis=1)
    return images

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
    cv.imshow("Detectat 1",img12) 
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

if __name__ == "__main__":
    flag1=0
    flag2=0
    imagine=input("Dati bancnota:")
    img=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Bancnote\\"+imagine+".jpg")
    cv.imshow("Color",img)
    img2=img.copy()
    h, w, c = img.shape
    imggray=ColorToGrayScale(img)
    cv.imshow('Alb-negru',imggray)
    img3=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Fig5.03_zgomotSarePiper.jpg")
    h1,w1,c1=img3.shape
    denoised1=Denoising(img3,w1,h1)
    cv.imshow('Fara zgomot',denoised1)
    binar = Binarizare(imggray)
    cv.imshow('Binar',binar)

    m=0
    n=0  
    img1=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Monezi\\monezi1.jpg")
    coins=[]
    coins=CoinDetection(img1,30,190)
    coins = np.uint16(np.around(coins))
    if coins is not None:
        coins = np.uint16(np.around(coins))
    n=0
    for pt in coins[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        n=n+1
        cv.circle(img1, (a, b), r, (0, 252, 0), 3)
  
        cv.circle(img1, (a, b), 1, (0, 0, 255), 3)
    cv.imshow("Detected Circle", img1)
    
    if n>1:
        flagd=0
        flags=0
        flagc=0
        if flag2 == 0:
            for i in cercLeu:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img2,template)
                if m == True:
                    flag2=1
                    print("Leu romanesc")
            
            for i in steagEuro:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img2,template)
                if m == 1:
                    flag2=1
                    print("Euro")
            
            for i in coroaneLire:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=1
                    print("Lire")
                    break
            for i in semneDolari:
                template=cv.imread("C:\\Anul3_Semestrul1\\PIMP\\Template\\"+i+".jpg",0)
                m=Recognition(img,template)
                if m == 1:
                    flag2=1
                    print("Dolar american")
                    break
        
        if flag2==0:
            rs1,rc1,rd1=RegionDivide(binar,w,h)
            if flagc == 0:
                if(rc1<1 and rd1<1 and rs1<1):
                    print("Rubla Ruseasca")
                    flags=1
                    flagc=1
                    flagd=1
                    flags=1
                    flag2=1
            if flagd ==0:
                if(rd1<2 and rc1<1 and rs1<1):
                    print("Yen japoznez")
                    flags=1
                    flagc=1
                    flagd=1
                    flags=1
                    flag2=1
    cv.waitKey(0)
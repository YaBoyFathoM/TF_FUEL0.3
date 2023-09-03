import cv2
from tensorflow import keras
import tkinter as tk
import numpy as np
from utils import show
from PIL import Image, ImageTk
class Camera():
    def __init__(self):
        self.root = tk.Tk()
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.lmain = tk.Label(self.root)
        self.lmain.grid()
        self.cap = cv2.VideoCapture(0)
        self.overlay=cv2.imread("ts.png",cv2.IMREAD_GRAYSCALE).astype(np.bool8)
        self.box=0
        if not self.cap.isOpened():
            self.lmain.config(text="Unable to open camera: please grant appropriate permission permissions plugin and relaunch", wraplength=self.lmain.winfo_screenwidth())
            self.root.mainloop()
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 672)
            self.cap.set(cv2.CAP_PROP_FPS,20)


    def refresh(self):
                    if 650>=self.box>=500:
                        self.root.quit()
                    ret, frame = self.cap.read()
                    if not ret:
                        self.lmain.after(0, self.refresh)
                        return
                    y=130
                    x=100
                    osx,osy=self.overlay.shape
                    self.overlay_color=100
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frameclip=cv2.adaptiveThreshold(cv2image[100:100+osx,150:150+osy,0],1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,15,20)
                    contours, _ = cv2.findContours(frameclip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cbclr="red"
                    cbtxt="Align ticket for identification"
                    Captured=False
                    for contour in contours:
                        c=int(cv2.contourArea(contour)/100)
                        approx = cv2.approxPolyDP(contour, 0.1* cv2.arcLength(contour,True), True)

                        if len(approx)==4 and c>=1000:
                            cv2.drawContours(frameclip, [approx], 0, (0,0,0), 15)
                            self.overlay_color=200
                            cbtxt="get a little bit closer so I can read it"
                        if len(approx)==4 and c>=2500:
                                #cv2.drawContours(frameclip, [approx], 0, (0,0,0), 5)
                                points = np.squeeze(contour)
                                y = points[:,1]
                                x = points[:,0]
                                (topy, topx) = (np.min(y), np.min(x))
                                (bottomy, bottomx) = (np.max(y), np.max(x))
                                cropped = frameclip[topy+10:bottomy-10, topx+10:bottomx-10]
                                self.overlay_color=255
                                cbclr="green"
                                cbtxt="                        CAPTURED                        "
                                tp=np.array(cropped).astype(np.int8)
                                if tp.shape[0]>=600 and tp.shape[1]>=400 and 1<=(tp.sum()/10000)<=4:
                                    self.tp=tp
                                    cv2.imwrite("tp.png",((tp*255)))
                                    self.root.quit()
                    capture_button=tk.Button(self.root,bg=cbclr, text=cbtxt,command=None)
                    capture_button.grid(row=1,column=0)
                    cv2image[100:100+osx,150:150+osy,1] = (frameclip+self.overlay)*self.overlay_color
                    w = self.lmain.winfo_screenwidth()
                    h = self.lmain.winfo_screenheight()
                    cw = cv2image.shape[0]
                    ch = cv2image.shape[1]
                    cw, ch = ch, cw
                    if (w > h) != (cw > ch):
                        cw, ch = ch, cw
                        cv2image = cv2.rotate(cv2image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    w = min(cw * h / ch, w)
                    h = min(ch * w / cw, h)
                    w, h = int(w), int(h)
                    cv2image = cv2.resize(cv2image, (w, h), interpolation=cv2.INTER_LINEAR)
                    img = Image.fromarray(cv2image)
                    self.imgtk = ImageTk.PhotoImage(image=img)
                    self.lmain.configure(image=self.imgtk)
                    self.lmain.update()
                    self.lmain.after(0, self.refresh)

photo=Camera()
photo.refresh()
photo.root.mainloop()
photo.root.destroy()
lines=cv2.imread("tswlines.png",0)
lines=cv2.resize(lines,(258,438))
class Ticket:
    def __init__(self) -> None:
        pass
    def align(self,im,tmp,orbs,tight):
        tpGray=cv2.imread(im,cv2.IMREAD_GRAYSCALE)
        templateGray=cv2.imread(tmp,cv2.IMREAD_GRAYSCALE)
        tpGray=cv2.resize(tpGray,(383,585))
        templateGray=cv2.resize(templateGray,(383,585))
        templateGray[147:,125:]=lines
        orb = cv2.ORB_create(orbs)
        (kpsA, descsA) = orb.detectAndCompute(tpGray, None)
        (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)
        matches = sorted(matches, key=lambda x:x.distance)
        keep = int(len(matches)*tight)
        matches = matches[:keep]
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        for (i, m) in enumerate(matches):
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
        (h, w) = templateGray.shape
        chars=(cv2.warpPerspective(tpGray, H, (w, h)))
        temp=templateGray.astype(np.float32)
        chars=chars-temp
        chars=chars[148:266,130:426]
        chars=np.where(chars>=1,chars,0)
        self.chars=chars
        self.truck=chars[110:130,265:280]#truck
        self.serial=chars[110:130,280:355]#serial
        self.m1=chars[4:32,:140]#m1
        self.m2=chars[4:32,140:280]#m2
        self.m1e=chars[32:60,:140]#m1e
        self.m2e=chars[32:60,140:280]#m2e
        self.g1=chars[60:88,:140]#g1
        self.g2=chars[60:88,140:280]#g2
        show(chars[32:60,140:280],1)
        self.all=(
        self.m1,self.m1e,
        self.m2,self.m2e,
        self.g1,self.g2)

tp="assets/tp.png"
template="assets/tswlabs.png"
lines=cv2.imread("assets/tswlines.png",0)
ocr=keras.models.load_model("assets/ocr")
ticket=Ticket()
ticket.align(tp,template,1000,tight=0.1)
def read():
    for field in ticket.all[:4]:
        charpreds=[]
        if field.shape!=(28,140):
            field=cv2.resize(field,(140,28))
        charpreds=ocr.predict(np.expand_dims(field,0))
        show(field,10,title=charpreds)
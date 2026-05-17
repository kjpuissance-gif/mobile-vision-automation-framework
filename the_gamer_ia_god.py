#!/usr/bin/env python3
# THE GAMER IA GOD v13.0 — L'ABSOLU (version allégée)
import subprocess, time, os, sys, argparse, threading, random, re, numpy as np
from collections import deque
import logging
from typing import Optional, List, Dict, Any

GROQ_AVAILABLE = CV2_AVAILABLE = GROUNDING_AVAILABLE = YOLO_AVAILABLE = OCR_AVAILABLE = False
try:
    from groq import Groq; GROQ_AVAILABLE = True
except: pass
try:
    import cv2; CV2_AVAILABLE = True
except: pass
try:
    from groundingdino.util.inference import load_model, predict; GROUNDING_AVAILABLE = True
except: pass
try:
    from ultralytics import YOLO; YOLO_AVAILABLE = True
except: pass
try:
    import easyocr; OCR_AVAILABLE = True
except: pass

class DivineLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)
    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def warning(self, msg): self.logger.warning(msg)

class RealADBController:
    def __init__(self, device_id=None):
        self.device_id = device_id
        self.logger = DivineLogger("ADB")
        self.width, self.height = self._get_resolution()
        self.logger.info(f"📱 Résolution: {self.width}x{self.height}")
    def _adb(self, *args, timeout=5.0):
        cmd = ["adb"] + (["-s", self.device_id] if self.device_id else []) + list(args)
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout, text=True).stdout
        except Exception as e:
            self.logger.error(f"ADB error: {e}")
            return ""
    def _get_resolution(self):
        try:
            out = self._adb("shell", "wm", "size")
            for line in out.split("\n"):
                if "Physical" in line or "Override" in line:
                    parts = line.split(":")[-1].strip().split("x")
                    return int(parts[0]), int(parts[1])
        except: pass
        return 1080, 2400
    def tap(self, x, y):
        x = max(0, min(self.width-1, int(x)))
        y = max(0, min(self.height-1, int(y)))
        self._adb("shell", "input", "tap", str(x), str(y))
        time.sleep(random.uniform(0.05, 0.1))
    def swipe(self, x1, y1, x2, y2, duration=100):
        self._adb("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration))
        time.sleep(0.1)
    def capture_screen(self, path="/tmp/screen.png"):
        try:
            self._adb("shell", "screencap", "-p", "/sdcard/screen.png")
            self._adb("pull", "/sdcard/screen.png", path)
            if CV2_AVAILABLE and os.path.exists(path):
                return cv2.imread(path)
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
        return None

class DivineVision:
    def __init__(self):
        self.logger = DivineLogger("Vision")
        self.grounding = None
        self.yolo = None
        self.ocr = None
        self.enemy_cache = []
        self.last_full = 0
        self.scale = 0.5
        if GROUNDING_AVAILABLE and CV2_AVAILABLE:
            try:
                self.grounding = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth", device="cpu")
                self.logger.info("✅ Grounding DINO chargé")
            except Exception as e: self.logger.warning(f"Grounding DINO fail: {e}")
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO("yolov8n.pt")
                self.logger.info("✅ YOLO chargé")
            except: pass
        if OCR_AVAILABLE:
            try:
                self.ocr = easyocr.Reader(['en'], gpu=False)
                self.logger.info("✅ EasyOCR chargé")
            except: pass
        if CV2_AVAILABLE:
            self.kalman = cv2.KalmanFilter(4,2)
            self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
            self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
            self.kalman.processNoiseCov = np.eye(4, dtype=np.float32)*0.03
        else: self.kalman = None

    def _preprocess(self, frame):
        return cv2.resize(frame, (0,0), fx=self.scale, fy=self.scale) if frame is not None else None

    def detect_enemies(self, frame, force=False):
        if frame is None: return []
        small = self._preprocess(frame)
        now = time.time()
        if self.grounding and (force or now - self.last_full > 30):
            try:
                boxes, logits, _ = predict(self.grounding, small, "enemy. person. soldier. opponent.", box_threshold=0.35, text_threshold=0.25, device="cpu")
                enemies = []
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = [int(v/self.scale) for v in box]
                    enemies.append({"x":x1,"y":y1,"w":x2-x1,"h":y2-y1,"center":((x1+x2)//2,(y1+y2)//2),"confidence":float(logits[i]),"area":(x2-x1)*(y2-y1)})
                if enemies:
                    self.enemy_cache, self.last_full = enemies, now
                    return enemies
            except Exception as e: self.logger.error(f"Grounding DINO error: {e}")
        if self.yolo:
            try:
                results = self.yolo(small, verbose=False)
                enemies = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:
                                x1,y1,x2,y2 = [int(v/self.scale) for v in box.xyxy[0].tolist()]
                                enemies.append({"x":x1,"y":y1,"w":x2-x1,"h":y2-y1,"center":((x1+x2)//2,(y1+y2)//2),"confidence":float(box.conf[0]),"area":(x2-x1)*(y2-y1)})
                if enemies:
                    self.enemy_cache = enemies
                    return enemies
            except: pass
        if self.enemy_cache and self.kalman:
            pred = []
            for e in self.enemy_cache:
                cx,cy = e["center"]
                self.kalman.correct(np.array([[np.float32(cx)],[np.float32(cy)]]))
                p = self.kalman.predict()
                ne = e.copy()
                ne["center"] = (int(p[0]), int(p[1]))
                pred.append(ne)
            return pred
        return self.enemy_cache if self.enemy_cache else self._skin_detection(frame)

    def _skin_detection(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0,20,70]), np.array([20,255,255]))
            enemies = []
            for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(c) > 500:
                    x,y,w,h = cv2.boundingRect(c)
                    enemies.append({"x":x,"y":y,"w":w,"h":h,"center":(x+w//2,y+h//2),"confidence":1.0,"area":w*h})
            return enemies
        except: return []

    def read_text(self, frame, region=None):
        if not self.ocr: return ""
        try:
            if region:
                h,w = frame.shape[:2]
                roi = frame[int(region[1]*h):int(region[3]*h), int(region[0]*w):int(region[2]*w)]
            else: roi = frame
            return " ".join(self.ocr.readtext(roi, detail=0, paragraph=True))
        except: return ""

    def detect_hp(self, frame):
        txt = self.read_text(frame, (0.02,0.02,0.25,0.08))
        if txt:
            nums = re.findall(r'\d+', txt)
            if nums and 0<=int(nums[0])<=100: return float(nums[0])
        if not CV2_AVAILABLE: return 100
        try:
            h,w = frame.shape[:2]
            roi = frame[int(h*0.02):int(h*0.08), int(w*0.02):int(w*0.25)]
            mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), np.array([35,100,100]), np.array([85,255,255]))
            ratio = np.count_nonzero(mask)/max(1,mask.size)
            return min(100, ratio*100)
        except: return 100

    def detect_ammo(self, frame):
        txt = self.read_text(frame, (0.75,0.02,0.98,0.08))
        if txt:
            nums = re.findall(r'\d+', txt)
            if nums: return min(100, int(nums[0])/3)
        if not CV2_AVAILABLE: return 100
        try:
            h,w = frame.shape[:2]
            roi = frame[int(h*0.02):int(h*0.08), int(w*0.75):int(w*0.98)]
            mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), np.array([15,100,100]), np.array([35,255,255]))
            ratio = np.count_nonzero(mask)/max(1,mask.size)
            return min(100, ratio*100)
        except: return 100

class DivineGameBot:
    def __init__(self, device_id=None):
        self.logger = DivineLogger("Bot")
        self.adb = RealADBController(device_id)
        self.vision = DivineVision()
        self.state = {"hp":100,"ammo":100,"enemies":[],"last_actions":deque(maxlen=10)}
        self.running = False
        self.user_order = None
        api_key = os.environ.get("GROQ_API_KEY") or (open(".api_key").read().strip() if os.path.exists(".api_key") else None)
        self.groq = Groq(api_key=api_key) if api_key and GROQ_AVAILABLE else None
        self.consciousness = None
        if self.groq:
            self.consciousness = type("", (), {
                "chat": lambda s,msg: self._groq_chat(msg),
                "decide_action": lambda s,state,order=None: self._groq_decision(state, order)
            })()
            self.logger.info("🧠 Conscience Groq active")
        else:
            self.logger.warning("Pas de Groq → mode règles simples")
        threading.Thread(target=self._listen, daemon=True).start()

    def _groq_chat(self, msg):
        try:
            comp = self.groq.chat.completions.create(model="llama3-8b-8192", messages=[{"role":"system","content":"Tu es un dieu bienveillant, parle français court."},{"role":"user","content":msg}], temperature=0.8, max_tokens=200)
            return comp.choices[0].message.content
        except: return "Erreur de connexion."

    def _groq_decision(self, state, order):
        prompt = f"HP={state['hp']:.0f}% Ammo={state['ammo']:.0f}% Ennemis={len(state['enemies'])}. Dernières actions={list(state['last_actions'])}. "
        if order: prompt += f"ORDRE: {order}. "
        prompt += "Réponds exactement au format ACTION: <HEAL|RELOAD|ENGAGE|PATROL|TAP x y|SWIPE x1 y1 x2 y2> RAISON: <courte>"
        try:
            comp = self.groq.chat.completions.create(model="llama3-8b-8192", messages=[{"role":"user","content":prompt}], temperature=0.5, max_tokens=60)
            resp = comp.choices[0].message.content
            action, reason = "PATROL", ""
            for line in resp.split("\n"):
                if line.startswith("ACTION:"): action = line.split(":",1)[1].strip().upper()
                elif line.startswith("RAISON:"): reason = line.split(":",1)[1].strip()
            if not any(action.startswith(v) for v in ["HEAL","RELOAD","ENGAGE","PATROL","TAP","SWIPE"]): action="PATROL"
            return action, reason
        except: return "PATROL", "Erreur"

    def _listen(self):
        while self.running:
            try:
                inp = input("\n💬 Vous: ").strip()
                if inp.lower() == "exit":
                    self.running = False
                    break
                if any(inp.lower().startswith(kw) for kw in ("tire","recharge","soigne","patrouille","engage","va","bouge","attaque","cache","saute","cours","swipe","tap")):
                    self.user_order = inp
                    print("✅ Ordre reçu")
                elif inp.lower().startswith(("je te présente","connais-tu")):
                    print("🤖 GOD: Je note ce jeu, merci.")
                else:
                    if self.groq:
                        print(f"🤖 GOD: {self._groq_chat(inp)}")
                    else:
                        print("🤖 GOD: Mode dégradé, pas de dialogue.")
            except: pass

    def _fallback_action(self):
        if self.state["hp"] < 30: return "HEAL", "HP bas"
        if self.state["ammo"] < 20: return "RELOAD", "Ammo faible"
        if len(self.state["enemies"]) > 0: return "ENGAGE", "Ennemi"
        return "PATROL", "Routine"

    def _execute(self, action, reason):
        self.logger.info(f"🎬 {action} – {reason}")
        w,h = self.adb.width, self.adb.height
        if action == "HEAL": self.adb.tap(int(w*0.1), int(h*0.9))
        elif action == "RELOAD": self.adb.tap(int(w*0.1), int(h*0.85))
        elif action == "ENGAGE":
            if self.state["enemies"]:
                t = max(self.state["enemies"], key=lambda e: e["area"])
                cx, cy = t["center"]
                self.adb.tap(cx, cy)
                time.sleep(0.1)
                self.adb.tap(int(w*0.9), int(h*0.85))
            else: self.adb.tap(random.randint(w//3,2*w//3), random.randint(h//3,2*h//3))
        elif action == "PATROL":
            self.adb.tap(random.randint(int(w*0.2),int(w*0.8)), random.randint(int(h*0.3),int(h*0.7)))
        elif action.startswith("TAP"):
            parts = action.split()
            if len(parts)>=3: self.adb.tap(int(parts[1]), int(parts[2]))
        elif action.startswith("SWIPE"):
            parts = action.split()
            if len(parts)>=5: self.adb.swipe(*map(int, parts[1:5]))
        self.state["last_actions"].append(action)

    def run(self):
        self.running = True
        self.logger.info("🚀 Dieu du jeu éveillé")
        cycle = 0
        while self.running:
            frame = self.adb.capture_screen()
            if frame is not None:
                self.state["enemies"] = self.vision.detect_enemies(frame, force=(cycle%10==0))
                self.state["hp"] = self.vision.detect_hp(frame)
                self.state["ammo"] = self.vision.detect_ammo(frame)
            self.logger.info(f"Cycle {cycle}: HP={self.state['hp']:.0f}% Ammo={self.state['ammo']:.0f}% Ennemis={len(self.state['enemies'])}")
            if self.user_order:
                if self.groq:
                    a,r = self.consciousness.decide_action(self.state, self.user_order)
                else:
                    a,r = self._fallback_action()
                self.user_order = None
            else:
                if self.groq:
                    a,r = self.consciousness.decide_action(self.state)
                else:
                    a,r = self._fallback_action()
            self._execute(a, r)
            cycle += 1
            time.sleep(0.6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="ADB device ID")
    args = parser.parse_args()
    print("""\n🧠 THE GAMER IA GOD v13.0 (slim) – Parle, ordonne, il joue tout seul.\n""")
    DivineGameBot(args.device).run()

if __name__ == "__main__":
    main()
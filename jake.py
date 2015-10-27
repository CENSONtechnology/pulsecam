import cv2, numpy, time
import requests

class FaceDetection(object):

    def __init__(self, frame):
        self.gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def detect_faces(self):
        faceCascade      = cv2.CascadeClassifier('C:\Users\Jake\Desktop\python\heart_rate\cascades\haarcascade_frontalface_default.xml')
        return faceCascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 50))

class trackFace(object):

    def __init__(self, fps):
        # Buffer
        self.buf_size = int(5 * 30) # Duration * FPS
        self.buf = []
        self.fps = 90
        self.blur = 150
        self.last_send = 0;

    def get_forehead(self, x, y, w, h):
        """
        Gets the forehead by looking a the top of the bounds, and in the center.
        """
        y = y 
        w = w / 2
        h = h / 6
        x = x + (w / 2)

        return tuple(map(int, (x,y,w,h)))

    def draw_forehead(self, frame, forehead):
        # DRAW PLZ
        c = (0,255,0)
        x, y, w, h = forehead
        cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), c, 2)

    def get_forehead_sample(self, frame, forehead):
        # Get an average of the green channel in on the forehead
        x, y, w, h = forehead
        img = frame[y:h+y, x:x+w]
        # Lets show the users forehead seperately
        cv2.imshow("Camera2", img)
        # Grab the mean colour of the green colours in the image
        sample = cv2.mean(img)[1]
        self.add_sample(sample)
        self.get_data()
        return sample

    @property        
    def buffer_full(self):
        """
        We don't want the buffer to get to big - so we'll check if its past its
        maximum size here.
        """
        return len(self.buf) >= self.buf_size

    def add_sample(self, value):
        frame_time = time.time()
        # Append the value of the sample to our list
        self.buf.append((frame_time, value))
        if self.buffer_full:
            # Remove from system
            self.buf.pop(0)

    def get_data(self):
        # Get the "ideal" evenly spaced times
        even_times = numpy.linspace(self.buf[0][0], self.buf[-1][0], len(self.buf))        
        # Interpolate the data to generate evenly temporally spaced samples
        interpolated = numpy.interp(even_times, *zip(*self.buf))        
        # Perform the FFT
        fft = numpy.fft.rfft(interpolated)
        fft = zip(numpy.abs(fft), numpy.angle(fft))
        # Best fit
        best_bin = fft[len(fft)-1][0]
        heartrate = self.bin_to_bpm(best_bin)
        r = requests.get("http://tracker.curtish.me/save?value=" + str(int(heartrate)))
        print('Pulse found: ' + str(int(test)))

    def bin_to_bpm(self, bin):
        return (bin * self.fps) / self.blur

    def bpm_to_bin(self, bpm):
        return int(float(len(self.buf) * bpm) / float(60.0 * 90.0))

# Create object to read images from camera 1 (my external webcam)
cam = cv2.VideoCapture(1)
fps = cam.get(cv2.CAP_PROP_FPS)
tracker = trackFace(fps);

while True:

    #Get image from webcam
    ret, frame = cam.read()

    # Find a face/s
    face = FaceDetection(frame)
    faces = face.detect_faces()

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # Draw face bounds
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Get Forehead
        forehead = tracker.get_forehead(x, y, w, h)
        tracker.draw_forehead(frame, forehead)
        # Get the sample and work on it
        tracker.get_forehead_sample(frame, forehead)
        
    #show the result
    cv2.imshow("Camera", frame)

    #Sleep infinite loop for ~10ms
    #Exit if user presses <Esc>
    if cv2.waitKey(10) == 27:
        break
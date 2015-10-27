## Extreme Medicine Hackathon
# PulseCam
 
---
 
### Contributors
 
* Jake Price
* Curtis Harding
 
---
 
Ability to track the pulse rate of multiple people using a webcam source. We did this by tracking users faces, taking there foreheads, converting the image to just its green chnnel and monitoring the changes in the optical absortion characteristcs of haemaglobin.

---
 
 
### How to run
 
Make sure you have [python 2.7] , [opencv] and [numpy] installed
 
```sh
git clone [repo url]
cd pulsecam
python jake.py
```
 
---
 
### Improvements
 
* Better camera feed
* Differenctiate between people
 
 
[python]:https://www.python.org/
[opencv]:http://opencv.org/
[numpy]:https://pypi.python.org/pypi/numpy
[repo url]:https://github.com/extrememedicine/pulsecam
**AI Face Recognition & Biometric Entry System**
A real-time browser-based face recognition application built using **HTML, CSS, and JavaScript** with **face-api.js**. The system uses the webcam to detect and recognize faces, register new users, and manage biometric profiles directly in the browser.
Features
* Real-time face detection using webcam
* Face recognition and user matching
* Multi-angle face enrollment
* Age and gender estimation
* Biometric profile registration and deletion
* LocalStorage-based user registry
* Live FPS and system status monitoring
* Responsive and modern glassmorphism UI
* Camera access and detection status indicators
* Low-latency face detection mode
Technologies Used
* HTML5
* CSS3
* JavaScript
* face-api.js
* WebRTC / MediaDevices API
* Browser LocalStorage
* Font Awesome
How It Works
The application loads pre-trained face recognition models, accesses the user's webcam, detects faces in real time, and generates face descriptors. Registered face descriptors are stored locally and compared with live camera input to identify matching users.
 **Note:** This project is intended for learning and demonstration purposes. Biometric data is stored locally in the browser and should not be considered suitable for production security without additional privacy and security measures.

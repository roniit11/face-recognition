/* ==========================================================================
   Biometrics Entry - Javascript Core Logic
   ========================================================================== */

// Model weights CDN
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

// State Management
let modelsLoaded = false;
let webcamActive = false;
let registeredUsers = [];
let faceMatcher = null;
let activeTab = 'pane-scan'; // active tab pane ID
let currentFPS = 0;
let lastFrameTime = performance.now();
let frameCount = 0;
let isLowLatencyMode = false;
let detectionLoopId = null;

// Multi-Angle Registration State
let enrollmentDescriptors = [];
let lastEnrollmentCaptureTime = 0;
const REQUIRED_ENROLLMENT_ANGLES = 8;
const ENROLLMENT_THROTTLE_MS = 300; // capture every 300ms to get different angles

// Temporary holder for biometric capture details
let tempCaptureData = null;

// Access Overlay States
let isAccessOverlayActive = false;
let cooldownUserPhone = null;
let cooldownTimeoutId = null;
let clearCooldownTimeoutId = null;

// DOM Elements
const webcamEl = document.getElementById('webcam');
const canvasEl = document.getElementById('overlay-canvas');
const systemStatusInd = document.getElementById('system-status-indicator');
const loadingOverlay = document.getElementById('loading-overlay');
const videoOverlayMsg = document.getElementById('video-overlay-msg');
const scanLine = document.getElementById('scan-line');
const fpsDisplay = document.getElementById('fps-display');
const btnToggleCamera = document.getElementById('btn-toggle-camera');
const btnSwitchDetector = document.getElementById('btn-switch-detector');
const toastEl = document.getElementById('toast');

// Overlay Elements
const accessOverlayEl = document.getElementById('access-overlay');
const accessNameEl = document.getElementById('access-name');
const accessTimeEl = document.getElementById('access-time');
const accessDemographicsEl = document.getElementById('access-demographics');
const btnCloseAccess = document.getElementById('btn-close-access');

// Initialize the Application
window.addEventListener('DOMContentLoaded', async () => {
    // 1. Load profiles from localStorage
    loadUserRegistry();

    // 2. Initialize Neural Models
    const success = await initNeuralModels();
    if (!success) return;

    // 3. Start Webcam Stream
    const camSuccess = await initWebcam();
    if (!camSuccess) return;

    // 4. Hide Loading Overlay and Start Loops
    hideLoadingOverlay();
    startDetectionLoop();

    // 5. Setup Action Listeners
    setupEventListeners();
});

// Update Loading Step UI
function updateLoadingStep(stepId, state) {
    const el = document.getElementById(stepId);
    if (!el) return;

    el.className = `step-${state}`;
    const icon = el.querySelector('.step-icon');
    
    if (state === 'active') {
        icon.className = 'fa-solid fa-circle-notch fa-spin step-icon';
    } else if (state === 'done') {
        icon.className = 'fa-solid fa-circle-check step-icon';
    } else if (state === 'error') {
        icon.className = 'fa-solid fa-circle-xmark step-icon';
    }
}

// Load face-api neural models
async function initNeuralModels() {
    try {
        updateLoadingStep('step-library', 'done');
        
        updateLoadingStep('step-detector', 'active');
        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        updateLoadingStep('step-detector', 'done');
        
        updateLoadingStep('step-landmarks', 'active');
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        updateLoadingStep('step-landmarks', 'done');
        
        updateLoadingStep('step-recognizer', 'active');
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        updateLoadingStep('step-recognizer', 'done');
        
        updateLoadingStep('step-age', 'active');
        await faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL);
        updateLoadingStep('step-age', 'done');
        
        modelsLoaded = true;
        updateSystemStatus('ready', 'Biometrics Entry Online');
        return true;
    } catch (err) {
        console.error("Error loading neural networks:", err);
        updateLoadingStep('step-detector', 'error');
        updateSystemStatus('error', 'Model Loading Failure');
        showToast("Failed loading neural models. See console.", "error");
        return false;
    }
}

// Initialize Webcam Sensor
async function initWebcam() {
    updateLoadingStep('step-camera', 'active');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        webcamEl.srcObject = stream;
        
        // Wait for video load
        await new Promise((resolve) => {
            webcamEl.onloadedmetadata = () => resolve();
        });
        
        updateLoadingStep('step-camera', 'done');
        webcamActive = true;
        
        // Remove text overlay message
        videoOverlayMsg.style.display = 'none';
        scanLine.style.display = 'block';
        
        return true;
    } catch (err) {
        console.error("Camera connection error:", err);
        updateLoadingStep('step-camera', 'error');
        updateSystemStatus('error', 'Sensor Feed Offline');
        videoOverlayMsg.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-error"></i><span>Camera Feed Blocked</span>';
        showToast("Camera access is blocked or unavailable.", "error");
        return false;
    }
}

// System Status Bar Update
function updateSystemStatus(state, text) {
    const dot = systemStatusInd.querySelector('.status-dot');
    const txtEl = systemStatusInd.querySelector('.status-text');
    
    dot.className = `status-dot status-${state}`;
    txtEl.textContent = text;
}

// Fade out loading screen
function hideLoadingOverlay() {
    loadingOverlay.style.opacity = '0';
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
    }, 500);
}

// Local Storage Registry Loader
function loadUserRegistry() {
    const stored = localStorage.getItem('auraface_registry');
    if (stored) {
        try {
            registeredUsers = JSON.parse(stored);
            updateFaceMatcher();
        } catch (e) {
            console.error("Corrupted database found, resetting:", e);
            registeredUsers = [];
            localStorage.setItem('auraface_registry', JSON.stringify([]));
        }
    }
    renderRegistryList();
}

// Re-compile LabeledFaceDescriptors for FaceMatcher
// Maps multiple descriptors per user to handle orientation & expression changes!
function updateFaceMatcher() {
    if (registeredUsers.length === 0) {
        faceMatcher = null;
        return;
    }
    
    const labeledDescriptors = registeredUsers.map(user => {
        // Retrieve multiple templates. Fallback to single descriptor if old profile format
        const descriptorsRaw = user.descriptors || [user.descriptor];
        const descFloat32Arrays = descriptorsRaw.map(d => new Float32Array(d));
        return new faceapi.LabeledFaceDescriptors(user.phone, descFloat32Arrays);
    });
    
    // Initialize matcher with Euclidean threshold of 0.6
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
}

// Render profiles grid
function renderRegistryList() {
    const grid = document.getElementById('registry-grid');
    const emptyState = document.getElementById('registry-empty');
    const countBadge = document.getElementById('registry-count');
    
    countBadge.textContent = `${registeredUsers.length} profile${registeredUsers.length !== 1 ? 's' : ''}`;
    
    if (registeredUsers.length === 0) {
        grid.style.display = 'none';
        emptyState.style.display = 'flex';
        return;
    }
    
    emptyState.style.display = 'none';
    grid.style.display = 'grid';
    grid.innerHTML = '';
    
    registeredUsers.forEach(user => {
        const card = document.createElement('div');
        card.className = 'registry-card';
        
        const genderVal = user.registeredGender || 'Unknown';
        
        card.innerHTML = `
            <img class="card-avatar" src="${user.avatar}" alt="${user.name}">
            <div class="card-info">
                <h4>${escapeHTML(user.name)}</h4>
                <span class="phone">${escapeHTML(user.phone)}</span>
                <span class="age-tag">${genderVal}, ${user.registeredAge} yrs</span>
            </div>
            <button class="btn-card-delete" title="Delete Profile" data-phone="${user.phone}">
                <i class="fa-regular fa-trash-can"></i>
            </button>
        `;
        
        // Delete button listener
        card.querySelector('.btn-card-delete').addEventListener('click', (e) => {
            e.stopPropagation();
            const phoneToDelete = e.currentTarget.getAttribute('data-phone');
            deleteProfile(phoneToDelete);
        });
        
        grid.appendChild(card);
    });
}

// Delete user profile
function deleteProfile(phone) {
    registeredUsers = registeredUsers.filter(u => u.phone !== phone);
    localStorage.setItem('auraface_registry', JSON.stringify(registeredUsers));
    updateFaceMatcher();
    renderRegistryList();
    showToast("Biometric profile removed.", "info");
}

// Face Detection Loop
function startDetectionLoop() {
    if (detectionLoopId) cancelAnimationFrame(detectionLoopId);
    
    const detectorOptions = isLowLatencyMode 
        ? new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 })
        : new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });

    async function runDetection() {
        if (!webcamActive || webcamEl.paused || webcamEl.ended) {
            detectionLoopId = requestAnimationFrame(runDetection);
            return;
        }

        // Measure FPS
        const now = performance.now();
        frameCount++;
        if (now - lastFrameTime >= 1000) {
            currentFPS = Math.round((frameCount * 1000) / (now - lastFrameTime));
            fpsDisplay.textContent = `${currentFPS} FPS`;
            frameCount = 0;
            lastFrameTime = now;
        }

        try {
            // Run detections with landmarks, descriptors, age, and gender
            const detections = await faceapi.detectAllFaces(webcamEl, detectorOptions)
                .withFaceLandmarks()
                .withFaceDescriptors()
                .withAgeAndGender();

            const ctx = canvasEl.getContext('2d');
            ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

            // Resize coordinates to viewport size
            const displaySize = { width: webcamEl.offsetWidth, height: webcamEl.offsetHeight };
            if (canvasEl.width !== displaySize.width || canvasEl.height !== displaySize.height) {
                canvasEl.width = displaySize.width;
                canvasEl.height = displaySize.height;
            }
            
            faceapi.matchDimensions(canvasEl, displaySize);
            const resizedDetections = faceapi.resizeResults(detections, displaySize);

            // Clear display logs if no faces
            if (resizedDetections.length === 0) {
                handleNoFacesDetected();
            } else {
                handleFacesDetected(resizedDetections);
            }
        } catch (e) {
            console.error("Frame processing error:", e);
        }

        // Run next frame loop (throttled to avoid freezing UI)
        setTimeout(() => {
            detectionLoopId = requestAnimationFrame(runDetection);
        }, 30); 
    }

    detectionLoopId = requestAnimationFrame(runDetection);
}

// Handle layout states when zero faces are in frame
function handleNoFacesDetected() {
    // 1. Scan console tab update
    const badge = document.getElementById('scan-badge');
    badge.className = 'badge';
    badge.textContent = 'NO SUBJECT';
    
    document.getElementById('match-confidence').textContent = '';
    document.getElementById('res-name').textContent = '—';
    document.getElementById('res-phone').textContent = '—';
    document.getElementById('res-gender').textContent = '—';
    document.getElementById('res-age').textContent = '—';
    document.getElementById('scan-result-avatar').innerHTML = '<i class="fa-solid fa-user-shield"></i>';
    
    // Disable registration triggers
    const consoleRegBtn = document.getElementById('btn-console-register');
    consoleRegBtn.classList.add('disabled');
    consoleRegBtn.disabled = true;

    // 2. Registration tab preview update
    if (activeTab === 'pane-register') {
        document.getElementById('register-avatar-canvas').style.display = 'none';
        document.getElementById('register-avatar-fallback').style.display = 'flex';
        document.getElementById('register-age-feedback').textContent = 'No face in frame...';
        
        // Reset enrollment progress bar if face leaves during enrollment
        if (enrollmentDescriptors.length < REQUIRED_ENROLLMENT_ANGLES) {
            enrollmentDescriptors = [];
            updateEnrollmentUI();
        }
        document.getElementById('btn-submit-registration').disabled = true;
    }
}

// Handle layout states when one or more faces are in frame
function handleFacesDetected(resizedDetections) {
    const ctx = canvasEl.getContext('2d');
    
    // Prioritize the largest bounding box in view
    let primaryDetection = resizedDetections[0];
    let maxArea = 0;
    
    resizedDetections.forEach(det => {
        const area = det.detection.box.width * det.detection.box.height;
        if (area > maxArea) {
            maxArea = area;
            primaryDetection = det;
        }
    });

    const now = performance.now();
    const ageEst = primaryDetection.age;
    
    // Format Gender nicely
    const rawGender = primaryDetection.gender || 'unknown';
    const genderEst = rawGender === 'male' ? 'Male' : rawGender === 'female' ? 'Female' : 'Non-binary';
    
    const currentDescriptor = primaryDetection.descriptor;

    // Crop avatar bounding box
    const videoSizeRatio = webcamEl.videoWidth / webcamEl.offsetWidth;
    const box = primaryDetection.detection.box;
    const cropBox = {
        x: box.x * videoSizeRatio,
        y: box.y * videoSizeRatio,
        width: box.width * videoSizeRatio,
        height: box.height * videoSizeRatio
    };
    const cropCanvas = cropFace(webcamEl, cropBox);
    const croppedAvatarDataUrl = cropCanvas.toDataURL('image/jpeg', 0.85);

    // Save temporary snapshot capture data
    tempCaptureData = {
        descriptor: Array.from(currentDescriptor),
        age: Math.round(ageEst),
        gender: genderEst,
        avatar: croppedAvatarDataUrl
    };

    // ========================================================
    // TAB 2: REGISTER PROFILE PROCESS (ROLLING DESCRIPTOR ENROLLMENT)
    // ========================================================
    if (activeTab === 'pane-register') {
        const regCanvas = document.getElementById('register-avatar-canvas');
        const regFallback = document.getElementById('register-avatar-fallback');
        const previewCtx = regCanvas.getContext('2d');
        
        regCanvas.style.display = 'block';
        regFallback.style.display = 'none';
        
        previewCtx.clearRect(0, 0, 120, 120);
        previewCtx.drawImage(cropCanvas, 0, 0, 120, 120);
        
        // Add descriptor to array if throttled timeframe passed
        if (enrollmentDescriptors.length < REQUIRED_ENROLLMENT_ANGLES) {
            if (now - lastEnrollmentCaptureTime >= ENROLLMENT_THROTTLE_MS) {
                enrollmentDescriptors.push(Array.from(currentDescriptor));
                lastEnrollmentCaptureTime = now;
                updateEnrollmentUI();
            }
        }
        
        // Display capture feedback
        if (enrollmentDescriptors.length < REQUIRED_ENROLLMENT_ANGLES) {
            document.getElementById('register-age-feedback').textContent = `Analyzing: ${genderEst}, ~${Math.round(ageEst)} yrs`;
        } else {
            document.getElementById('register-age-feedback').textContent = `Face captured: ${genderEst}, ~${Math.round(ageEst)} yrs`;
        }
        
        validateRegistrationForm();
    }

    // ========================================================
    // TAB 1: ACTIVE SCAN LOGIC & ACCESS OVERLAY TRIGGERS
    // ========================================================
    let matchedProfile = null;
    let confidenceText = "";
    let distance = 1.0;

    if (faceMatcher && currentDescriptor) {
        const bestMatch = faceMatcher.findBestMatch(currentDescriptor);
        if (bestMatch.label !== 'unknown') {
            matchedProfile = registeredUsers.find(u => u.phone === bestMatch.label);
            distance = bestMatch.distance;
            // Map Euclidean distance to match confidence percentage
            const confidence = Math.max(0, Math.min(99, Math.round((1 - distance) * 200)));
            confidenceText = `${confidence}% Match`;
        }
    }

    // Trigger overlay if matched, not active, and outside active cooldowns
    if (matchedProfile && activeTab === 'pane-scan' && !isAccessOverlayActive) {
        if (cooldownUserPhone !== matchedProfile.phone) {
            triggerAccessOverlay(matchedProfile, ageEst);
        }
    }

    if (activeTab === 'pane-scan') {
        const badge = document.getElementById('scan-badge');
        const resAvatar = document.getElementById('scan-result-avatar');
        
        if (matchedProfile) {
            badge.className = 'badge recognized';
            badge.textContent = 'ACCESS GRANTED';
            
            document.getElementById('match-confidence').textContent = confidenceText;
            document.getElementById('res-name').textContent = matchedProfile.name;
            document.getElementById('res-phone').textContent = matchedProfile.phone;
            document.getElementById('res-gender').textContent = matchedProfile.registeredGender || 'Unknown';
            document.getElementById('res-age').textContent = `${Math.round(ageEst)} yrs (Registered: ${matchedProfile.registeredAge} yrs)`;
            
            resAvatar.innerHTML = `<img src="${matchedProfile.avatar}" alt="${matchedProfile.name}">`;
            
            // Disable register trigger since profile already exists
            const consoleRegBtn = document.getElementById('btn-console-register');
            consoleRegBtn.classList.add('disabled');
            consoleRegBtn.disabled = true;
        } else {
            badge.className = 'badge unknown';
            badge.textContent = 'UNKNOWN FACE';
            
            document.getElementById('match-confidence').textContent = '';
            document.getElementById('res-name').textContent = 'Unknown Subject';
            document.getElementById('res-phone').textContent = 'No Biometric Record';
            document.getElementById('res-gender').textContent = genderEst;
            document.getElementById('res-age').textContent = `${Math.round(ageEst)} years`;
            
            resAvatar.innerHTML = `<img src="${croppedAvatarDataUrl}" alt="Unknown">`;
            
            // Enable register trigger
            const consoleRegBtn = document.getElementById('btn-console-register');
            consoleRegBtn.classList.remove('disabled');
            consoleRegBtn.disabled = false;
        }
    }

    // ========================================================
    // RENDER BIOMETRIC CAMERA BBOXES & LABELS
    // ========================================================
    resizedDetections.forEach(det => {
        const { x, y, width, height } = det.detection.box;
        
        let isPrimaryMatched = false;
        let detName = "Unknown";
        
        // Find match for this specific face index
        if (faceMatcher && det.descriptor) {
            const match = faceMatcher.findBestMatch(det.descriptor);
            if (match.label !== 'unknown') {
                const user = registeredUsers.find(u => u.phone === match.label);
                if (user) {
                    detName = user.name;
                    isPrimaryMatched = true;
                }
            }
        }

        // Draw HUD Corner Brackets
        ctx.strokeStyle = isPrimaryMatched ? 'var(--color-success)' : 'var(--color-primary)';
        ctx.lineWidth = 3.5;
        const bLen = Math.min(22, width * 0.18);
        
        // Top-Left
        ctx.beginPath(); ctx.moveTo(x, y + bLen); ctx.lineTo(x, y); ctx.lineTo(x + bLen, y); ctx.stroke();
        // Top-Right
        ctx.beginPath(); ctx.moveTo(x + width - bLen, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + bLen); ctx.stroke();
        // Bottom-Left
        ctx.beginPath(); ctx.moveTo(x, y + height - bLen); ctx.lineTo(x, y + height); ctx.lineTo(x + bLen, y + height); ctx.stroke();
        // Bottom-Right
        ctx.beginPath(); ctx.moveTo(x + width - bLen, y + height); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width, y + height - bLen); ctx.stroke();

        // Target bounding frame
        ctx.strokeStyle = isPrimaryMatched ? 'rgba(16, 185, 129, 0.2)' : 'rgba(99, 102, 241, 0.2)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);

        // HUD text tag
        ctx.fillStyle = isPrimaryMatched ? 'rgba(16, 185, 129, 0.85)' : 'rgba(99, 102, 241, 0.85)';
        ctx.font = '600 12px var(--font-heading)';
        
        const textLabel = `${detName} (${genderEst.substring(0, 1)}, ${Math.round(det.age)}y)`;
        const labelWidth = ctx.measureText(textLabel).width;
        
        // Label Background
        ctx.beginPath();
        ctx.roundRect(x, y - 24, labelWidth + 18, 19, [4, 4, 0, 0]);
        ctx.fill();
        
        ctx.fillStyle = '#ffffff';
        ctx.fillText(textLabel, x + 9, y - 10);
    });
}

// Update enrollment progress details on UI
function updateEnrollmentUI() {
    const bar = document.getElementById('enrollment-progress');
    const text = document.getElementById('enrollment-text');
    
    const pct = (enrollmentDescriptors.length / REQUIRED_ENROLLMENT_ANGLES) * 100;
    bar.style.width = `${pct}%`;
    
    if (enrollmentDescriptors.length < REQUIRED_ENROLLMENT_ANGLES) {
        text.textContent = `Capturing: ${enrollmentDescriptors.length}/${REQUIRED_ENROLLMENT_ANGLES} angles. Tilt head slightly...`;
        text.style.color = 'var(--color-secondary)';
    } else {
        text.textContent = `Capture completed (${REQUIRED_ENROLLMENT_ANGLES}/${REQUIRED_ENROLLMENT_ANGLES} angles). Enter details.`;
        text.style.color = 'var(--color-success)';
    }
}

// Utility to crop square avatar from video frame
function cropFace(video, crop) {
    const canvas = document.createElement('canvas');
    canvas.width = 120;
    canvas.height = 120;
    const ctx = canvas.getContext('2d');
    
    // Clamp coordinates to video frame boundaries
    let sx = Math.max(0, crop.x);
    let sy = Math.max(0, crop.y);
    let sw = Math.min(video.videoWidth - sx, crop.width);
    let sh = Math.min(video.videoHeight - sy, crop.height);
    
    // Crop a square aspect ratio if possible
    const size = Math.min(sw, sh);
    sx = sx + (sw - size) / 2;
    sy = sy + (sh - size) / 2;
    
    ctx.drawImage(video, sx, sy, size, size, 0, 0, 120, 120);
    return canvas;
}

// Dynamic form validator
function validateRegistrationForm() {
    const nameInput = document.getElementById('reg-name');
    const phoneInput = document.getElementById('reg-phone');
    const submitBtn = document.getElementById('btn-submit-registration');
    
    const isFormFilled = nameInput.checkValidity() && phoneInput.checkValidity();
    const isEnrollmentFinished = enrollmentDescriptors.length === REQUIRED_ENROLLMENT_ANGLES;
    
    submitBtn.disabled = !(isFormFilled && isEnrollmentFinished);
}

// Synthesize biometric success chime (C5 -> E5) using Web Audio API
function playSuccessBeep() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        const playTone = (freq, startTime, duration) => {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, startTime);
            
            gain.gain.setValueAtTime(0.12, startTime);
            gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
            
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            
            osc.start(startTime);
            osc.stop(startTime + duration);
        };
        
        const now = audioCtx.currentTime;
        playTone(523.25, now, 0.15); // C5 Tone
        playTone(659.25, now + 0.1, 0.3); // E5 Tone
    } catch (e) {
        console.error("Audio Synthesis Error:", e);
    }
}

// Trigger Full-Screen Access Granted modal
function triggerAccessOverlay(user, currentAge) {
    isAccessOverlayActive = true;
    
    // Configure overlay information
    accessNameEl.textContent = user.name;
    accessTimeEl.textContent = new Date().toLocaleTimeString();
    
    const gender = user.registeredGender || 'Unknown';
    const age = Math.round(currentAge);
    accessDemographicsEl.textContent = `${gender}, ${age} yrs`;
    
    // Play chime
    playSuccessBeep();
    
    // Display Modal
    accessOverlayEl.classList.add('show');
    
    // Cancel any pending cooldown clearers
    if (cooldownTimeoutId) clearTimeout(cooldownTimeoutId);
    if (clearCooldownTimeoutId) clearTimeout(clearCooldownTimeoutId);
    
    // Setup immediate cooldown user flag
    cooldownUserPhone = user.phone;
    
    // Auto-dismiss after 4 seconds
    cooldownTimeoutId = setTimeout(() => {
        hideAccessOverlay();
    }, 4000);
}

// Hide Access overlay and initiate loop cooldowns
function hideAccessOverlay() {
    accessOverlayEl.classList.remove('show');
    
    if (cooldownTimeoutId) clearTimeout(cooldownTimeoutId);
    
    // Set matching release cooldowns
    isAccessOverlayActive = false;
    
    // Keep user cooldown for 6 seconds to prevent double triggers of same person
    clearCooldownTimeoutId = setTimeout(() => {
        cooldownUserPhone = null;
    }, 6000);
}

// Setup Dashboard User Event Listeners
function setupEventListeners() {
    // 1. Tab switches
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            const targetPaneId = e.currentTarget.getAttribute('data-tab');
            if (!targetPaneId) return;

            // Update tab button classes
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            e.currentTarget.classList.add('active');

            // Switch content panes
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            document.getElementById(targetPaneId).classList.add('active');
            
            activeTab = targetPaneId;
            
            // Reset enrollment data if switching tab panels
            enrollmentDescriptors = [];
            updateEnrollmentUI();
            tempCaptureData = null;
        });
    });

    // 2. "Register Bounding Face" helper button in active scan
    document.getElementById('btn-console-register').addEventListener('click', () => {
        if (!tempCaptureData) return;
        
        // Switch to registration pane
        document.getElementById('tab-register').click();
    });

    // 3. Live Form input validation triggers
    const nameInput = document.getElementById('reg-name');
    const phoneInput = document.getElementById('reg-phone');
    
    nameInput.addEventListener('input', validateRegistrationForm);
    phoneInput.addEventListener('input', validateRegistrationForm);

    // 4. Form Submit Listener (Save User)
    document.getElementById('register-form').addEventListener('submit', (e) => {
        e.preventDefault();
        
        if (enrollmentDescriptors.length < REQUIRED_ENROLLMENT_ANGLES) {
            showToast(`Wait for biometric enrollment to capture 8 angles.`, "error");
            return;
        }

        const name = nameInput.value.trim();
        const phone = phoneInput.value.trim();

        // Check if phone number already exists
        const exists = registeredUsers.some(u => u.phone === phone);
        if (exists) {
            showToast("A user with this mobile number already exists.", "error");
            return;
        }

        // Save biometric profile with multiple descriptors
        const newUser = {
            name: name,
            phone: phone,
            registeredAge: tempCaptureData.age,
            registeredGender: tempCaptureData.gender,
            descriptors: enrollmentDescriptors, // Multiple descriptors!
            avatar: tempCaptureData.avatar
        };

        registeredUsers.push(newUser);
        localStorage.setItem('auraface_registry', JSON.stringify(registeredUsers));

        // Re-compile matcher
        updateFaceMatcher();
        renderRegistryList();

        // Clear fields
        nameInput.value = '';
        phoneInput.value = '';
        enrollmentDescriptors = [];
        updateEnrollmentUI();
        tempCaptureData = null;
        
        showToast("Biometric profile successfully registered!", "success");
        
        // Go back to scanning tab
        document.getElementById('tab-scan').click();
    });

    // 5. Toggle Camera Stream Pause/Play
    btnToggleCamera.addEventListener('click', () => {
        if (webcamActive) {
            webcamEl.pause();
            webcamActive = false;
            btnToggleCamera.innerHTML = '<i class="fa-solid fa-video"></i> Resume Feed';
            videoOverlayMsg.innerHTML = '<i class="fa-solid fa-pause"></i><span>Camera Paused</span>';
            videoOverlayMsg.style.display = 'flex';
            scanLine.style.display = 'none';
            
            // Clear drawing canvas
            const ctx = canvasEl.getContext('2d');
            ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
            handleNoFacesDetected();
        } else {
            webcamEl.play();
            webcamActive = true;
            btnToggleCamera.innerHTML = '<i class="fa-solid fa-video-slash"></i> Pause Feed';
            videoOverlayMsg.style.display = 'none';
            scanLine.style.display = 'block';
        }
    });

    // 6. Switch Latency Detector Modes
    btnSwitchDetector.addEventListener('click', () => {
        isLowLatencyMode = !isLowLatencyMode;
        
        if (isLowLatencyMode) {
            btnSwitchDetector.innerHTML = '<i class="fa-solid fa-shield-halved"></i> High Accuracy Mode';
            showToast("Switched to Low Latency processing (Tiny Face Net).", "info");
        } else {
            btnSwitchDetector.innerHTML = '<i class="fa-solid fa-bolt"></i> Low Latency Mode';
            showToast("Switched to High Accuracy processing (SsdMobilenet).", "info");
        }
        
        // Restart detection loop with new configurations
        startDetectionLoop();
    });

    // 7. Wipe Registry Database
    document.getElementById('btn-clear-all').addEventListener('click', () => {
        if (confirm("Are you sure you want to wipe all registered biometric signatures? This action cannot be undone.")) {
            registeredUsers = [];
            localStorage.setItem('auraface_registry', JSON.stringify([]));
            updateFaceMatcher();
            renderRegistryList();
            showToast("Database successfully cleared.", "info");
        }
    });

    // 8. Close Overlay Manual Trigger
    btnCloseAccess.addEventListener('click', () => {
        hideAccessOverlay();
    });
}

// Toast Notifications display helper
function showToast(message, type = 'success') {
    // Clear existing classes
    toastEl.className = 'toast';
    
    // Add type style
    toastEl.classList.add(`toast-${type}`);
    
    // Set message & icon
    const msgEl = document.getElementById('toast-message');
    const iconEl = document.getElementById('toast-icon');
    
    msgEl.textContent = message;
    
    if (type === 'success') {
        iconEl.className = 'fa-solid fa-circle-check toast-icon';
    } else if (type === 'error') {
        iconEl.className = 'fa-solid fa-circle-exclamation toast-icon';
    } else {
        iconEl.className = 'fa-solid fa-circle-info toast-icon';
    }
    
    // Show toast
    toastEl.classList.add('show');
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        toastEl.classList.remove('show');
    }, 3200);
}

// HTML XSS Escaping Helper
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

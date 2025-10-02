document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("uploadForm");
    const statusDiv = document.getElementById("status");

    // Handle form submission
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById("imageInput");
            const file = fileInput.files[0];

            if (!file) {
                statusDiv.innerText = "దయచేసి ఒక చిత్రం ఎంచుకోండి.";
                return;
            }

            statusDiv.innerText = "అనుమానం చేయబడుతోంది... దయచేసి వేచి ఉండండి.";

            const formData = new FormData();
            formData.append("file", file);

            try {
                const response = await fetch("http://127.0.0.1:8000/predict", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    window.location.href = "error.html";
                    return;
                }

                const data = await response.json();

                sessionStorage.setItem("result", JSON.stringify(data));
                window.location.href = "results.html";
            } catch (error) {
                console.error(error);
                window.location.href = "error.html";
            }
        });
    }

    // Handle results page
    if (window.location.href.includes("results.html")) {
        const resultData = JSON.parse(sessionStorage.getItem("result"));

        if (!resultData) {
            window.location.href = "index.html";
        } else {
            document.getElementById("diseaseName").innerText = "రోగం: " + resultData.disease;
            document.getElementById("confidence").innerText = "నమ్మకం: " + (resultData.confidence * 100).toFixed(2) + "%";
            document.getElementById("advice").innerText = "సలహా: " + resultData.advice;

            speakTelugu(resultData.advice);
        }
    }

    function speakTelugu(text) {
        if ("speechSynthesis" in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = "te-IN"; // Telugu language
            utterance.rate = 0.9;
            utterance.pitch = 1;
            speechSynthesis.speak(utterance);
        } else {
            console.log("Text-to-Speech not supported in this browser.");
        }
    }
});

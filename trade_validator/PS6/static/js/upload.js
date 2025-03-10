// Document upload functionality
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadList = document.getElementById('uploadList');
    const startUpload = document.getElementById('startUpload');
    const uploadStatus = document.getElementById('uploadStatus');
    let files = [];

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('highlight');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('highlight');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('highlight');
        const droppedFiles = e.dataTransfer.files;
        handleFiles(droppedFiles);
    });

    // Handle file input
    fileInput.addEventListener('change', (e) => {
        const selectedFiles = e.target.files;
        handleFiles(selectedFiles);
    });

    function handleFiles(files) {
        for (const file of files) {
            const fileItem = document.createElement('div');
            fileItem.classList.add('file-item');
            fileItem.textContent = file.name;
    
            // Add a remove button (optional)
            const removeButton = document.createElement('button');
            removeButton.textContent = 'Remove';
            removeButton.addEventListener('click', () => {
                uploadList.removeChild(fileItem);
                // Handle file removal on server-side if needed
            });
            fileItem.appendChild(removeButton);
    
            uploadList.appendChild(fileItem);
    
            // Send the file to the server-side script (e.g., using Fetch API or XMLHttpRequest)
            const formData = new FormData();
            formData.append('file', file);
    
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                // Handle response from server
                console.log(response);
            })
            .catch(error => {
                console.error(error);
            });
        }
    }

    // Handle file removal
    uploadList.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove-file')) {
            const index = parseInt(e.target.dataset.index);
            files.splice(index, 1);
            updateFileList();
            startUpload.disabled = files.length === 0;
        }
    });

    // Handle upload
    startUpload.addEventListener('click', () => {
        uploadStatus.hidden = false;
        simulateUpload();
    });

    function simulateUpload() {
        let progress = 0;
        const progressBar = uploadStatus.querySelector('.progress');
        const statusText = uploadStatus.querySelector('.status-text');

        const interval = setInterval(() => {
            progress += 5;
            progressBar.style.width = `${progress}%`;
            
            if (progress < 50) {
                statusText.textContent = 'Uploading files...';
            } else if (progress < 80) {
                statusText.textContent = 'Processing documents...';
            } else {
                statusText.textContent = 'Finalizing...';
            }

            if (progress >= 100) {
                clearInterval(interval);
                statusText.textContent = 'Upload complete!';
                setTimeout(() => {
                    window.location.href = 'validation-results.html';
                }, 1000);
            }
        }, 100);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
});
// Dashboard functionality
document.addEventListener('DOMContentLoaded', () => {
    // Simulate user data (in a real app, this would come from an API)
    const userData = {
        name: 'John Doe',
        recentDocuments: [
            { id: 1, name: 'Invoice-2024-001.pdf', status: 'Validated' },
            { id: 2, name: 'Shipping-Doc-123.pdf', status: 'Pending' },
            { id: 3, name: 'Certificate-456.pdf', status: 'Validated' }
        ]
    };

    // Update user name
    document.getElementById('userName').textContent = userData.name;

    // Populate recent documents
    const documentList = document.getElementById('recentDocuments');
    if (documentList) {
        userData.recentDocuments.forEach(doc => {
            const docElement = document.createElement('div');
            docElement.className = 'document-item';
            docElement.innerHTML = `
                <span>${doc.name}</span>
                <span class="status ${doc.status.toLowerCase()}">${doc.status}</span>
            `;
            documentList.appendChild(docElement);
        });
    }

    // Handle logout
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            // In a real app, this would clear session/tokens
            window.location.href = '../index.html';
        });
    }
});
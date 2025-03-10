// Authentication related JavaScript
document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            // Here you would typically make an API call to authenticate
            console.log('Login attempt:', { email, password });
            
            // For demo purposes, redirect to dashboard
            window.location.href = 'dashboard.html';
        });
    }
});
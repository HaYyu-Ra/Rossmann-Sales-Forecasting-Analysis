// Wait for the DOM to fully load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Rossmann Sales Forecasting Project - Static Files Loaded');

    // Example: Add dynamic interaction to a button
    const forecastButton = document.getElementById('forecastButton');
    if (forecastButton) {
        forecastButton.addEventListener('click', () => {
            alert('Forecasting data is being processed. Please wait...');
        });
    }

    // Example: Display today's date in a footer
    const footerDate = document.getElementById('footerDate');
    if (footerDate) {
        const today = new Date();
        footerDate.textContent = `Today's Date: ${today.toDateString()}`;
    }
});

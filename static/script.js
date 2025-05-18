document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const priceDollar = document.getElementById('price-dollar');
    const priceBirr = document.getElementById('price-birr');
    const timestamp = document.getElementById('timestamp');

    // Populate year dropdown
    const yearSelect = document.getElementById('year');
    const currentYear = new Date().getFullYear();
    const startYear = 1990; // Reasonable starting year for used cars

    // Create year groups
    const yearGroups = [
        { label: 'New Cars (Last 5 years)', start: currentYear - 4, end: currentYear },
        { label: 'Recent Cars (5-10 years)', start: currentYear - 9, end: currentYear - 5 },
        { label: 'Older Cars (10-20 years)', start: currentYear - 19, end: currentYear - 10 },
        { label: 'Classic Cars (Before ' + (currentYear - 19) + ')', start: startYear, end: currentYear - 20 }
    ];

    // Add year options with groups
    yearGroups.forEach(group => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = group.label;

        // Add years in descending order
        for (let year = group.end; year >= group.start; year--) {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            optgroup.appendChild(option);
        }

        yearSelect.appendChild(optgroup);
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
        submitButton.disabled = true;
        
        try {
            const formData = new FormData(form);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                priceDollar.textContent = data.prediction_dollar;
                priceBirr.textContent = data.prediction_birr;
                timestamp.textContent = `Last updated: ${data.timestamp}`;
                resultDiv.style.display = 'block';
                
                // Smooth scroll to result
                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            // Restore button state
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
        }
    });

    // Add input validation for kilometers driven
    const kmsInput = document.getElementById('kms_driven');
    kmsInput.addEventListener('input', function() {
        if (this.value < 0) {
            this.value = 0;
        }
    });

    // Add nice select styling
    const selects = document.querySelectorAll('.form-select');
    selects.forEach(select => {
        select.addEventListener('change', function() {
            if (this.value) {
                this.classList.add('selected');
            } else {
                this.classList.remove('selected');
            }
        });
    });
}); 
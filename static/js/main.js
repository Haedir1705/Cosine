/**
 * Main JavaScript for Thesis Similarity Detector
 */

document.addEventListener('DOMContentLoaded', function() {
    // Highlight active navigation link based on current page
    highlightActiveNavLink();

    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        initTooltips();
    }

    // Initialize form validation for all forms with .needs-validation class
    initFormValidation();

    // Add animation effects
    addAnimationEffects();

    // Initialize range sliders with tooltips
    initRangeSliders();

    // Initialize similarity visualization progress bars if present
    initSimilarityProgressBars();
});

/**
 * Highlight the active navigation link based on current page
 */
function highlightActiveNavLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(function(link) {
        const href = link.getAttribute('href');

        // Exact match
        if (href === currentPath) {
            link.classList.add('active');
        }
        // Check if current path starts with the link href (for nested routes)
        else if (href !== '/' && currentPath.startsWith(href)) {
            link.classList.add('active');
        }
    });
}

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validation for all forms with .needs-validation class
 */
function initFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');

    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }

            form.classList.add('was-validated');
        }, false);
    });
}

/**
 * Add animation effects to elements
 */
function addAnimationEffects() {
    // Add fade-in effect to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.classList.add('fade-in');
        card.style.animationDelay = (index * 0.1) + 's';
    });

    // Add hover effects to buttons (already in CSS, but could add more here)
}

/**
 * Initialize range sliders with tooltips
 */
function initRangeSliders() {
    const rangeInputs = document.querySelectorAll('input[type="range"]');

    rangeInputs.forEach(input => {
        const valueDisplay = document.getElementById(input.id + '_value');

        if (valueDisplay) {
            // Update value display on input
            input.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
            });
        }
    });
}

/**
 * Copy text to clipboard
 * @param {string} text - The text to copy
 * @returns {boolean} - Whether the copy was successful
 */
function copyToClipboard(text) {
    // Create temporary textarea
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';  // Avoid scrolling to bottom
    document.body.appendChild(textarea);
    textarea.select();

    try {
        // Execute copy command
        const successful = document.execCommand('copy');
        document.body.removeChild(textarea);
        return successful;
    } catch (err) {
        console.error('Failed to copy text: ', err);
        document.body.removeChild(textarea);
        return false;
    }
}

/**
 * Show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - The type of toast (success, error, warning, info)
 * @param {number} duration - Duration in milliseconds
 */
function showToast(message, type = 'info', duration = 3000) {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');

    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }

    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-${type} text-white">
                <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    // Initialize and show the toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: duration });
    toast.show();

    // Remove toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * Toggle dark mode
 */
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');

    // Save preference to localStorage
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
}

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

/**
 * Initialize and configure similarity progress bars
 */
function initSimilarityProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');

    progressBars.forEach(bar => {
        // Get the value
        const value = parseFloat(bar.getAttribute('aria-valuenow'));

        // Apply the appropriate class based on the value
        if (value >= 80) {
            bar.classList.add('bg-danger');
        } else if (value >= 70) {
            bar.classList.add('bg-warning');
            bar.classList.add('text-dark');
        } else if (value >= 60) {
            bar.classList.add('bg-primary');
        } else {
            bar.classList.add('bg-secondary');
        }

        // Fix any styling issues
        bar.style.display = 'flex';
        bar.style.alignItems = 'center';
        bar.style.justifyContent = 'center';
        bar.style.fontWeight = '500';
    });
}

// Export functions for external use
window.thesisApp = {
    copyToClipboard: copyToClipboard,
    showToast: showToast,
    toggleDarkMode: toggleDarkMode
};

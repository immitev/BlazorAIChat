import DOMPurify from './lib/dompurify/dist/purify.es.mjs';
import * as marked from './lib/marked/dist/marked.esm.js';

const purify = DOMPurify(window);

customElements.define('assistant-message', class extends HTMLElement {
    static observedAttributes = ['markdown'];

    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'markdown') {

            // Remove <citation> tags
            newValue = newValue.replace(/<citation.*?<\/citation>/gs, '');

            // Parse the markdown to HTML
            const elements = marked.parse(newValue);

            // Sanitize the HTML
            const sanitizedElements = purify.sanitize(elements, { KEEP_CONTENT: false });

            // Escape HTML code blocks
            const escapedHtml = sanitizedElements.replace(/<code>(.*?)<\/code>/gs, (match, p1) => {
                return `<code>${p1.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code>`;
            });

            // Set the innerHTML
            this.innerHTML = escapedHtml;

        }
    }
});



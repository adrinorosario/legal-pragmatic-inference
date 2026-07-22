const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    try {
        console.log('Launching browser...');
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        
        const filePath = path.resolve(__dirname, 'book.html');
        console.log(`Loading file: file://${filePath}`);
        
        await page.goto(`file://${filePath}`, { waitUntil: 'networkidle0' });
        
        const outputPath = path.resolve(__dirname, 'Data_Engineering_Book.pdf');
        console.log(`Generating PDF to: ${outputPath}`);
        
        await page.pdf({
            path: outputPath,
            format: 'A4',
            printBackground: true,
            displayHeaderFooter: false,
            margin: {
                top: '0px',
                bottom: '0px',
                left: '0px',
                right: '0px'
            }
        });
        
        console.log('PDF generated successfully!');
        await browser.close();
    } catch (err) {
        console.error('Error generating PDF:', err);
        process.exit(1);
    }
})();

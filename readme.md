## Plagiarism Detector

This is a Python application that uses Flask, pandas, matplotlib, sklearn, and nltk to detect plagiarism in text files. The application allows users to upload multiple text files and generates a report indicating the similarity between each pair of files.

The application is hosted at [https://plagiarism-detector-dst6.onrender.com/](https://plagiarism-detector-dst6.onrender.com/)

### Features

- Upload multiple text files for comparison
- Generate a similarity report in Excel format
- Visualize similarity between files with a heatmap

### How to Run

The application is containerized using Docker and hosted on Render. You can access it directly through the browser.

### Usage

- On the home page, click "Choose Files" to select the text files you want to compare.
- Click "Upload" to upload the files and start the comparison.
- Once the comparison is complete, a heatmap showing the similarity between each pair of files will be displayed.
- Click "Download Report" to download an Excel file containing the similarity report.

### Future Improvements

- Add the ability to choose between different language models (MLE and WBI) for comparison and visualize perplexity.
- Add a tag cloud with the most common words in the text.
- Add charts showing the dependency of the similarity measure on the length of the text.
- Compare computation time for different language models.
- Add charts showing the ratio of the number of words in the text to the entire text for each word and all texts on one chart (histogram) with the ability to choose the text for comparison (dropdown or radio buttons).
- Add the ability to dynamically add and remove files for comparison.
- Add text boxes for pasting compared texts.
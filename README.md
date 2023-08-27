# Image Search Code

This repository contains Python code for an image search application built using Streamlit and CLIP (Contrastive Language-Image Pretraining). The application allows users to search for images within a given collection using natural language queries. It leverages the CLIP model to generate image and text embeddings, enabling efficient similarity-based searches.

[streamlit-app-2023-08-27-00-08-13.webm](https://github.com/iamprakhargupta/Image_search_app/assets/30393146/b2bdcb7e-d482-4864-b4bd-2074f262a346)



## Features

- Search for images in a collection using natural language queries.
- Utilizes the CLIP model to encode images and text into embeddings.
- Displays search results with images ranked by similarity to the search query.
- Adjustable settings for the number of search results and image display width.

## Getting Started

Follow these steps to set up and run the image search application on your local machine:

1. **Clone the Repository:**

    ```sh
    git clone https://github.com/yourusername/image-search-app.git
    cd image-search-app
    ```

2. **Install Dependencies:**

    Ensure you have Python and pip installed. Run the following command to install the required packages:

    ```sh
    pip install streamlit torch clip
    ```

3. **Run the Application:**

    Run the Streamlit app using the following command:

    ```sh
    streamlit run app.py
    ```

4. **Use the Application:**

    - Enter the path to the directory containing your collection of images in the provided input field.
    - Enter a search query in the text input field, describing the image you're looking for.
    - Adjust the settings on the sidebar, such as the number of search results and image display width.
    - View the search results displayed in a responsive grid layout.
  
## Adjusting Settings

- Use the sidebar sliders to adjust the number of search results and the width at which the images are displayed.

## Contributing

Contributions to this project are welcome! If you find any issues or have ideas for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
   

# UC_Merced_classification  

Mobile net based classification for UC merced data   
# Accuracy metrics
Accuracy: 0.97
# Model Outputs
![Model output](https://github.com/der-knight/UC_Merced_classification/blob/main/helper/output.png)

## Getting data
Go to http://weegee.vision.ucmerced.edu/datasets/landuse.html
## Data Directory
Create a new folder named data with with images in folder structure (.// data // UCMerced_LandUse // UCMerced_LandUse// Images //* // *.tif)
## File Structure
# The helper folder contains io_tools and viz_tools.
1) The io_tools subsets the data to get 14 classes and subset the data into training validation and testing in 70:10:20 ratio. The io_tools also has a function thatloads data as normalized array
2) Viz_tools helps exploratory data analysis and also helps plot misclassified images to identify algorithm efficacy

'''
this is an ImageJ macro written in Jython. It receives brightfield microscope images of a single organoid in each image as inputs, it processes the images to generate masks that represent the organoid outline and area while reducing signals from the background such as debris or shades. It then uses the generated ROIs to generate a morphometric profile for each individual organoid and for whole datasets of organoids.

- make sure there are no empty subfolder within the target folder（to fix, exclude empty folders）

MIT License

Copyright (c) 2024 Edgar A. Galan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# Import required packages
import os, sys, math, datetime, time, re
from ij import IJ
from ij import ImagePlus
from ij import WindowManager
from ij import ImageStack
from ij.io import DirectoryChooser
from ij.measure import ResultsTable
from ij.measure import Measurements
from ij.process import ImageProcessor
from ij.process import ImageConverter
from ij.gui import WaitForUserDialog, GenericDialog, Overlay, ShapeRoi
from ij.plugin.frame import RoiManager
from ij.plugin.filter import ParticleAnalyzer
import shutil
from java.awt import Color #for line color
from ij.process import ByteProcessor #to draw ROI mask

# List of supported image extensions
supported_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif')

# Function to create or clear directories
def create_subdirectories(root):
    roi_subfolder_directory = os.path.join(root, "ROI")
    masks_subfolder_directory = os.path.join(root, "masks")
    failed_measurements_directory = os.path.join(masks_subfolder_directory, "failed_measurements")

    for directory in [roi_subfolder_directory, masks_subfolder_directory, failed_measurements_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            clear_directory(directory)

    return roi_subfolder_directory, masks_subfolder_directory, failed_measurements_directory

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


minimum_size = 1000 # for fluorescence, excludes single cells, low values make all ROIs be discarded
maximum_size = 10000000 # do not matter, we quantify whathever area is stained

area_threshold = 33000 #for criteria to exclude single cells or small pieces or organoids ,this value is ok for 20x, there is a problem with this, higher values make roim be empty, apparently all ROIs are discarded
pixel_size = 0.86 # my microscope
pixel_size = 1 # values are given in pixels

minimum_circularity = 0 # for particle analysis, if you trust your thresholds keep this same as threshold, lower it to check if your thrheshold is discarding good organoids or set to low value or zero to just measure everything and curate manually later, only ROIs within minimums will be saved
circularity_threshold = 0.1 #for fluorescence we measure all signal in the ROI, we are not concerned about the ROI not being circular

# Define the buffer size as a percentage of the image dimensions
edge_margin = 0.1  # Adjust as needed to exclude edge particles
edge_margin = 0.1  # for whole organoid fluorescence we will actually exclude edges because they are always at the center

# Get input and output directories with GUI
dc = DirectoryChooser("Choose an input directory")     
inputDirectory = dc.getDirectory() 

dc = DirectoryChooser("Choose an output directory")
outputDirectory = dc.getDirectory()

# Regular expressions
re_day = r'd(\d+)'  # Matches 'd' followed by one or more digits
#re_day = r'd(\d{1,3})'  # Matches 'd' followed by one to three digits
#re_day = r'd(\d{1,2})'  # Matches 'd' followed by one or two digits
re_last_numbers = r'\d+(?=\.[^.]+$)'  # Matches the last series of consecutive numbers

re_org = r'org(\d+)' # match one or more digits that follow the string "org", this is the organoid number ID

# Initialize an empty dictionary to store day-organoid-area data
day_organoid_area_map = {}
# Initialize an empty dictionary to store areas for day 0 organoids
day0_area_map = {}
mean_area_day0 = 0

# Path to save the log file
log_filename = os.path.basename(inputDirectory.rstrip('\\')) + "_log.txt"
log_file_path = os.path.join(outputDirectory, log_filename)

# Open the log file
log_file = open(log_file_path, "w")


# Redirect stdout to log file and ImageJ console
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure output is written immediately

    def flush(self):
        for f in self.files:
            f.flush()

# Save the original stdout
original_stdout = sys.stdout

# Redirect stdout to log file and ImageJ console
sys.stdout = Tee(sys.stdout, log_file)


# Create the output CSV file
#output_filename = "output_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + ".csv"
output_filename = os.path.basename(inputDirectory.rstrip('\\')) + "_raw.csv"
output_path = os.path.join(outputDirectory, output_filename)


with open(output_path, "w") as output:
    # set the output column names for the CSV sheet
    output.write("Subfolder,File name,Day,Organoid number,Area,Corrected Total Fluorescence,Corrected Mean Fluorescence\n")
    
# Use os.walk to find subfolders in the input directory
    for root, dirs, files in os.walk(inputDirectory):
        # Check if the directory should be processed
        if "ROI" not in root and "mask" not in root and "ignore" not in root:
            roi_subfolder_directory, masks_subfolder_directory, failed_measurements_directory = create_subdirectories(root)

            # Process each file in the current directory
            for filename in files:
                image_path = os.path.join(root, filename)
                subfolder = os.path.relpath(root, inputDirectory)

                # Check if the file has a supported extension before opening
                if filename.lower().endswith(supported_extensions):
                    imp = IJ.openImage(image_path)
                    if imp:
                        print('analyzing: ' + filename)
                        # getting day and organoid number from filename using regex
                        re_day_match = re.search(re_day, filename)
                        # re_org_match = re.search(re_org, filename)
                        re_org_match = re.search(re_last_numbers, filename)
                        day = re_day_match.group(1).zfill(2) if re_day_match else "NA"
                        # organoid = re_org_match.group().zfill(3) if re_org_match else "NA"
                        organoid = re_org_match.group().zfill(3) if re_org_match else "NA"
                        print("day: " + str(day))
                        print("organoid: " + str(organoid))
                        # Get the dimensions of the image
                        pixel_width_x = imp.getWidth()
                        pixel_height_y  = imp.getHeight()
                        #print("x = " + str(x))
                        #print("x = " + str(y))
                        # Create a rectangle that excludes the outer border of the image
                        imp.setRoi(2, 2, pixel_width_x - 2, pixel_height_y - 2) #rectangle corner starts at 2,2 coordinate, we cut 2 pixels from each border of the image 
                        IJ.run(imp, "Crop", "") # Crop the image to get rid of possible line of white pixels surrounding the each image (it can affect exclued edges, because ROIs will never touch the edge)
                        scaled_width_x = (imp.getWidth() * pixel_size) # maybe unnecessary, but mathematically correct, just to be superprecise and recover those 4 pixels we cropped
                        scaled_height_y = (imp.getHeight() * pixel_size)
                        IJ.run(imp, "Set Scale...", "distance=1 known=" +str(pixel_size)+ " pixel=1 unit=um global")
                        IJ.run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area fraction redirect=None decimal=12");
                        # morphological operations
                        ic = ImageConverter(imp)
                        ic.convertToGray8()
                        IJ.setAutoThreshold(imp, "Default dark") #for fluorescence we use dark as we measure bright in dark background
                        # IJ.setAutoThreshold(imp, "Otsu")
                        IJ.run(imp, "Convert to Mask", "")
                        # IJ.run(imp, "Invert", "") #no invert for otsu because it measures black object on white background by default, using default also no invert
                        IJ.run(imp, "Fill Holes", "")
                        for i in range(25):
                            IJ.run(imp, "Dilate", "")
                        IJ.run(imp, "Fill Holes", "")
                        # for i in range(5):
                            # IJ.run(imp, "Erode", "")
                        # IJ.run(imp, "Fill Holes", "")
                        IJ.run(imp, "Gaussian Blur...", "sigma=10")
                        IJ.setAutoThreshold(imp, "Default dark")
                        # IJ.setAutoThreshold(imp, "Otsu dark")
                        IJ.run(imp, "Convert to Mask", "")
                        table = ResultsTable()
                        roim = RoiManager(True)
                        ParticleAnalyzer.setRoiManager(roim)
                        p = ParticleAnalyzer
                        pa = ParticleAnalyzer(p.ADD_TO_MANAGER | p.OVERLAY | p.SHOW_OVERLAY_OUTLINES | p.SHOW_MASKS, Measurements.AREA | Measurements.FERET | Measurements.CIRCULARITY | Measurements.SHAPE_DESCRIPTORS | Measurements.CENTROID | Measurements.PERIMETER | Measurements.ELLIPSE | Measurements.CENTER_OF_MASS | Measurements.INTEGRATED_DENSITY, table, minimum_size, maximum_size, minimum_circularity, 1.0)
                        # we will not use intden and median of particle analyzer because the measurement is taken from the mask, rather than grayscale, dividing intden by area we will see that the value is always 255, median is always 255, skewness and kurtosis cant be measured here because its all 255 value flat mask
                        #centroid and center of mass are the same
                        #feret angle and ellipe angle are not useful because there is no fixed frame of reference for organoids 
                        pa.setHideOutputImage(True)
                        pa.analyze(imp)
                        
                        
                        # #only do this if something is measured, saves calculatios
                        # print("x = " + str(scaled_width_x))
                        # print("y = " + str(scaled_height_y)) 
                        # edge_x = scaled_width_x * edge_margin
                        # edge_y = scaled_height_y * edge_margin
                        # print(edge_x)
                        # print(edge_y)
                        #print(table.getHeadings())
                        #table.show("Table Results")
                        print("table size after particle analysis = " + str(table.size()))
                        
                        # get a composite ROI of all ROIs so we can get the background intensity of anything that is not cells of organoids, otherwise floating cells fluorescence is considered as background, raising background value slightly, not really significant, but not strictly correct 
                        if roim is not None:
                            print("roim is not None, creating composite ROI")
                            # Get all ROIs from the ROI Manager
                            rois = roim.getRoisAsArray()

                            if len(rois) > 0:
                                print("len(rois) = " + str(len(rois)))
                                # Create a composite ROI from the first ROI
                                composite_roi = ShapeRoi(rois[0])

                                # Combine all other ROIs into the composite ROI
                                for roi in rois[1:]:
                                    composite_roi.or(ShapeRoi(roi))
                                print("composite_roi = " + str(composite_roi))
                                imp = IJ.openImage(image_path)            
                                IJ.run(imp, "Set Scale...", "distance=1 known=" +str(pixel_size)+ " pixel=1 unit=um global")
                                ImageConverter(imp).convertToGray8()
                                # Set the composite ROI as the active ROI on the image
                                imp.setRoi(composite_roi)
                                IJ.run(imp, "Make Inverse", "")
                                stats_bg = imp.getStatistics()
                                background_mean_intensity = stats_bg.mean
                                print(" background_mean_intensity = " + str(background_mean_intensity))
                        else:
                            output.write(str(subfolder) + ',' + filename + ',' + str(day) + ',' + str(organoid) + ",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA")
                            print(filename + " nothing was measured")
                            print("-" * 20)
                            # Save the output image generated by the Particle Analyzer in the "masks" directory for postprocessing human inspection, we can then see if our measurement failed, why was that
                            output_image_path = os.path.join(failed_measurements_directory, filename.replace(".", "_mask."))
                            output_image = imp.duplicate()
                            # Split the filename to get the root and extension
                            file_root, file_ext = os.path.splitext(filename)
                            output_image_path = os.path.join(failed_measurements_directory, "{}_mask.png".format(file_root))
                            output_image = imp.duplicate()
                            #output_image.setTitle("Output Image")
                            # output_image.setRoi(roim.getRoi(index))
                            IJ.saveAs(output_image, "PNG", output_image_path)

                        # after getting the intensity on the area that does not contain ROIs (rois can include single cells floating around), then we will delete ROIs smaller than threshold (single cells), so we get only data from the organoid, otherwise we have two options, we get data from all ROIs, or we count those single cells as part of the background, but if they are too many they will influence ctf calculations, later add to the ROI overlay with a different color all those ROIs that were not considered as background nor as organoid 
                        # Check if the ROI Manager exists
                        if roim:
                            print("len(rois) = " + str(len(rois)))
                            # Get the number of ROIs in the ROI Manager
                            roiCount = roim.getCount()
                            # Iterate over each ROI in the ROI Manager
                            for i in range(roiCount):
                                # Get the ROI at the current index
                                roi = roim.getRoi(i)                           
                                # Get the index of the ROI
                                index = roim.getRoiIndex(roi)
                                # Get the size of the ROI
                                size = roi.getStatistics().area
                                # Print the ROI index and size
                                print("ROI Index:", index)
                                print("ROI Size:", size)
                                print("-" * 20)
                                # IJ.log('ROI Index: %d' % index)
                                # IJ.log('ROI size: %d' % size)
                                # IJ.log('-' * 20)
                                    
                                    
                                    
                                    
# we do not need the table, but will use the table to delete the ROIs by their index
                        if table.size() != 0:                        
                            filtered_composite_roi = ShapeRoi(rois[0]) #maybe not the best approach, creates ROI with ROI index 0, then below we have to remove  the ROI 0, we could find a way to create the composite empty
                            filtered_composite_roi.not(ShapeRoi(rois[0]))
                            # Find the ROI with the largest area that does not touch the image boundaries
                            index = -1
                            maxArea = -1
                            column_area = table.getColumnIndex("Area")
                            # Check if Column  exists (in case it didn't measure anything)
                            #print("index area = " + str(column_area))
                            # if column_area != -1:# and table.getValue("Area", column_area) != 0: this check getvalue area can fail if there are no ROIs, so there is no value area
                                # print("pass")
                                # Find the ROI with the largest area that does not touch the image boundaries
                            for i, area in enumerate(table.getColumn(column_area)):
                                roi = roim.getRoi(i)
                                if area > area_threshold:
                                    index = i
                                    filtered_composite_roi.or(ShapeRoi(roi)) 
                                    print("ROI added to composite ROI " + str(i))
                                    # print("table size = " + str(table.size()))
                        print("table size after removing small particles = " + str(table.size())) #remember roim and table are different objects so they may have different quantity of entries
                        print("len(rois) = " + str(len(rois)))
                                    #print("index area to measure = " + str(index))
                                                         
                                    
                                    
                        print("filtered composite ROI created, composite_roi = " + str(composite_roi))
                        if roim is not None:
                            print("filtered_composite_roi = " + str(filtered_composite_roi))
                            print("ROI is ready for measurement")
                            composite_roi.not(filtered_composite_roi) #subtract measured ROI from all ROIs, for visualization purpuses only, otherwise composite ROI is drawn on ROI images and may overlap with the measured filter ROI over the organoid contour
                        imp = IJ.openImage(image_path)            
                        IJ.run(imp, "Set Scale...", "distance=1 known=" +str(pixel_size)+ " pixel=1 unit=um global")
                        ImageConverter(imp).convertToGray8()
                           
                                
                        # Get the dimensions of the original image
                        width, height = imp.getWidth(), imp.getHeight()
                        # Set the composite ROI as the active ROI on the image
                        imp.setRoi(filtered_composite_roi)
                        stats_roi = imp.getStatistics()
                        roi_mean_intensity = stats_roi.mean
                        roi_area = stats_roi.area
                        # roi_min = stats_roi.min
                        # roi_max = stats_roi.max
                        #add std dev and mode
                        
                        # Create a blank image with the same dimensions
                        blank_processor = ByteProcessor(width, height)
                        blank_processor.setColor(0)  # Set background to black
                        blank_processor.fill()  # Fill with black

                        # Draw the ROI onto the blank image
                        blank_processor.setColor(255)  # Set draw color to white
                        blank_processor.fill(filtered_composite_roi)  # Fill the ROI with white

                        # Create an ImagePlus object from the blank_processor
                        mask_imp = ImagePlus("Mask", blank_processor)

                        # Ensure the mask is in binary format
                        IJ.run(mask_imp, "8-bit", "")  # Convert to 8-bit if necessary
                        IJ.setThreshold(mask_imp, 1, 255)  # Set threshold to make mask binary
                        IJ.run(mask_imp, "Convert to Mask", "")  # Convert to mask

                        file_root, file_ext = os.path.splitext(filename)
                        output_image_path = os.path.join(masks_subfolder_directory, "{}_mask.png".format(file_root))
                        IJ.saveAs(mask_imp, "PNG", output_image_path)
                       
                        

                        ctf = (roi_area * roi_mean_intensity) - (roi_area * background_mean_intensity)
                        cmf = ctf / roi_area #corrected mean fluorescence
                        output.write(str(subfolder) + ',' + filename + ',' + day + ',' + organoid + ',' +  str(roi_area) + ',' + str(ctf) + ',' + str(cmf))

                        imp = IJ.openImage(image_path) 
                        # Ensure the image is in RGB format before drawing
                        #if not imp.getType() == ij.ImagePlus.COLOR_RGB:
                        IJ.run(imp, "RGB Color", "") #you can only draw color in rgb images, make sure images are rgb
                        # Get the ROI from the ROI Manager
                        #print("index for draw = " + str(index))
                        # roim.select(composite_roi)
                        # Make sure the ROI is actively selected on the image
                        
                        
                        imp.setRoi(composite_roi)
                        # Get the bounding rectangle of the ROI
                        bounds = composite_roi.getBounds()
                        print(bounds)
                        # Check if the bounding rectangle has x=0, y=0, width=0, and height=0
                        if bounds.width != 0 and bounds.height != 0: # if composite is same as filtered (no particles were excluded, that is no artifacts or debri, just a single roi) then composite will be 0 because is equals to filtered (nothing filtered) and therefore we have nothing to draw, bounds.x != 0 and bounds.y != 0 and  not necessary
                            # Set the drawing color, e.g., to red
                            print("composite_roi width = " + str(bounds.width))
                            IJ.setForegroundColor(0, 255, 255); # RGB values for cyan color
                            # Draw the ROI onto the image
                            IJ.run(imp, "Line Width...", "line=11");
                            IJ.run(imp, "Draw", "")
                        imp.setRoi(filtered_composite_roi)
                        # Set the drawing color, e.g., to red
                        IJ.setForegroundColor(255, 0, 255); # RGB values for magenta color
                        # Draw the ROI onto the image
                        IJ.run(imp, "Line Width...", "line=11");
                        IJ.run(imp, "Draw", "")
                        # Save the modified image with ROI in the "ROI" subdirectory as .tif
                        file_root, file_ext = os.path.splitext(filename)
                        output_image_path = os.path.join(roi_subfolder_directory, "{}_roi.png".format(file_root))
                        IJ.saveAs(imp, "PNG", output_image_path)
                        # roi_filename = filename.replace(".", "_ROI.")
                        # IJ.saveAs(imp, "PNG", os.path.join(roi_subfolder_directory, roi_filename))#, "compression=JPEG quality=50") #use tiff if you wish to preserve image information, e.g., pixel size or ROIs, but it will take much more storage space
                        print(filename + " - success")
                        # IJ.log('{} - success'.format(filename))
                        print("-" * 20)
                        
                    output.write('\n')
                    imp.changes = False                          
                    imp.close()
                    roim.reset()
                    roim.close()
# Close the CSV file after processing all subfolders
output.close()    
print("finished")
# IJ.log('finished')

# for sending to imagej console
# Close the log file
log_file.close()

# Reset stdout to original stdout
sys.stdout = original_stdout

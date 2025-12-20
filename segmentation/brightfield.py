'''
this is an ImageJ macro written in Jython. It receives brightfield microscope images of a single organoid in each image as inputs, it processes the images to generate masks that represent the organoid outline and area while reducing signals from the background such as debris or shades. It then uses the generated ROIs to generate a morphometric profile for each individual organoid and for whole datasets of organoids.

- make sure there are no empty subfolder within the target folder （to fix, exclude empty folders）

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
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os, math, datetime, time, re
from ij import IJ
from ij import ImagePlus
from ij import WindowManager
from ij import ImageStack
from ij.io import DirectoryChooser
from ij.measure import ResultsTable
from ij.measure import Measurements
from ij.process import ImageProcessor
from ij.process import ImageConverter
from ij.gui import WaitForUserDialog, GenericDialog, Overlay
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


minimum_size = 60000 # smallest organoid in medium group 38
maximum_size = 20000000 # largest organoid in medium group


area_threshold = maximum_size + 500000 #for criteria, we will let them be the same for the moment
# pixel_size = 2.994011976047904 #lab b
#pixel_size = 1.73, 0.86 my microscope
pixel_size = 0.86 # our microscope, 4x
# pixel_size = 1 # my microscope

#pixel_size = 0.86
# pixel_size = 2.99

minimum_circularity = 0.28 # for particle analysis, if you trust your thresholds keep this same as threshold, lower it to check if your thrheshold is discarding good organoids or set to low value or zero to just measure everything and curate manually later, only ROIs within minimums will be saved
circularity_threshold = 0.31 # for criteria, 0.3 is ok for lung organoid

# minimum and maximum organoid sizes in micrometer, the sqrt is because the value we input is um but imagej will use pixel, so we converted to pixel here
# minimum_size_um = 100000 / math.sqrt(pixel_size) #180000 something close to that number, smallest org of brain dataset

# Define the buffer size as a percentage of the image dimensions
# edge_margin = 0.28  # lung organoid 
edge_margin = 0.2  # wang jie brain organoids 2.4 px, it was ok


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
    output.write("Subfolder,File name,Day,Organoid number,Area,Growth rate,Perimeter,Feret maximum,Feret minumum,Ellipse major axis,Ellipse minor axis,Aspect ratio,Equivalent circle diameter,Circularity,Roundness,Solidity,Corrected total intensity,Corrected mean intensity,Integrated intensity,Mean intensity,Intensity standard deviation,Mode intensity,Median intensity,Corrected minimum intensity,Corrected maximum intensity,Skewness,Kurtosis,Centroid to center of mass distance,Criteria\n")
    
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
                        IJ.setAutoThreshold(imp, "Default")
                        # IJ.setAutoThreshold(imp, "Otsu")
                        IJ.run(imp, "Convert to Mask", "")

                        # IJ.run(imp, "Invert", "") #no invert for otsu because it measures black object on white background by default, using default also no invert
                        IJ.run(imp, "Fill Holes", "")
                        for i in range(4):
                            IJ.run(imp, "Dilate", "")
                        IJ.run(imp, "Fill Holes", "")
                        for i in range(5):
                            IJ.run(imp, "Erode", "")
                        IJ.run(imp, "Fill Holes", "")
                        IJ.run(imp, "Gaussian Blur...", "sigma=6.4")
                        IJ.setAutoThreshold(imp, "Default dark")
                        # IJ.setAutoThreshold(imp, "Otsu dark")
                        IJ.run(imp, "Convert to Mask", "")
                        # IJ.run(imp, "Watershed", "") # uncomment if multiple organoids in image                    
                        table = ResultsTable()
                        roim = RoiManager(True)
                        ParticleAnalyzer.setRoiManager(roim)
                        p = ParticleAnalyzer
                        pa = ParticleAnalyzer(p.ADD_TO_MANAGER | p.OVERLAY | p.EXCLUDE_EDGE_PARTICLES | p.SHOW_OVERLAY_OUTLINES | p.SHOW_MASKS, Measurements.AREA | Measurements.FERET | Measurements.CIRCULARITY | Measurements.SHAPE_DESCRIPTORS | Measurements.CENTROID | Measurements.PERIMETER | Measurements.ELLIPSE | Measurements.CENTER_OF_MASS, table, minimum_size, maximum_size, minimum_circularity, 1.0)
                        #centroid and center of mass are the same
                        #feret angle and ellipe angle are not useful because there is no fixed frame of reference for organoids 
                        pa.setHideOutputImage(True)
                        pa.analyze(imp)
                        
                        #only do this if something is measured, saves calculatios
                        print("x = " + str(scaled_width_x))
                        print("y = " + str(scaled_height_y)) 
                        edge_x = scaled_width_x * edge_margin
                        edge_y = scaled_height_y * edge_margin
                        print(edge_x)
                        print(edge_y)
                        #print(table.getHeadings())
                        #table.show("Table Results")
                        print("table size after particle analysis = " + str(table.size()))
                        # IJ.log('ROIs detected after particle analysis = %d' % table.size())
                        
                        # for i in range(table.size()): # Iterate over the particle analysis results table
                            # # Print the row index and the values in the row
                            # # print("Row index:", i)
                            # for column_name in table.getHeadings():
                                # value = table.getValue(column_name, i)
                                # print(column_name + ":", value)
                            # print("-" * 20)  # Add a separator between rows

                        # Check if the ROI Manager exists
                        if roim:
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
                        excluded_indices = []
                        print("table before removing edge particles = " + str(table.size()))
                        # IJ.log('ROIs before removing edge particles = %d' % table.size())
                        if table.size() != 0: #you can add else run particle analysis without minimum size or circ so that you can save the mask and see why measurement failed
                            
                            

                            # Exclude particles with centroid near the edges, they can touch the edge, as long as centroid is not near edge, sa that organoids that extended and touched the edge are not excluded, but shades that originate near edges are
                            
                            for i in range(table.size()):
                                # print("i = " + str(i))
                                # print("table size = " + str(table.size()))
                                centroid_x = table.getValue("X", i)
                                # print("centroid x = " + str(centroid_x))
                                centroid_y = table.getValue("Y", i)
                                # print(centroid_y)
                                # Check if the particle is within the buffer region
                                # print("centroid x before removing particles = " + str(centroid_x))
                                if centroid_x < edge_x or centroid_x > (scaled_width_x - edge_x) or centroid_y < edge_y or centroid_y > (scaled_height_y - edge_y):
                                    excluded_indices.append(i)
                                    # print("excluded " + str(excluded_indices))

                        # Remove excluded particles from the results table
                        # print("centroid x before removing particles 2 = " + str(centroid_x))
                        for index in reversed(excluded_indices):
                            table.deleteRow(index)
                            roim.select(index)
                            roim.runCommand("Delete")
                            # print("row deleted")
                            # print("table size = " + str(table.size()))
                            print("table size after removing edge particles = " + str(table.size()))
                            if roim: # Check if the ROI Manager exists
                                roiCount = roim.getCount() # Get the number of ROIs in the ROI Manager
                                for i in range(roiCount): # Iterate over each ROI in the ROI Manager
                                    roi = roim.getRoi(i) # Get the ROI at the current index
                                    index = roim.getRoiIndex(roi) # Get the index of the ROI
                                    size = roi.getStatistics().area # Get the size of the ROI
                                    # Print the ROI index and size
                                    #print("ROI Index:", index)
                                    #print("ROI Size:", size)
                                    #print("----------------------------------------")
                        # print("table size after removing edge particles = " + str(table.size()))
                        # print("centroid x after removing particles = " + str(centroid_x))
                        if table.size() != 0:                        
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
                                if area > maxArea:
                                    index = i
                                    print("max area is = " + str(area))
                                    maxArea = area
                                    #print("index area to measure = " + str(index))
                        else:
                            output.write(str(subfolder) + ',' + filename + ',' + str(day) + ',' + str(organoid) + ",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA")
                            print(filename + " nothing was measured")
                            print("-" * 20)
                            #break
                            # Save the output image generated by the Particle Analyzer in the "masks" directory for postprocessing human inspection, we can then see if our measurement failed, why was that
                            # output_image_path = os.path.join(failed_measurements_directory, filename.replace(".", "_mask."))
                            # output_image = imp.duplicate()
                            # Split the filename to get the root and extension
                            file_root, file_ext = os.path.splitext(filename)
                            output_image_path = os.path.join(failed_measurements_directory, "{}_mask.png".format(file_root))
                            output_image = imp.duplicate()
                            #output_image.setTitle("Output Image")
                            # output_image.setRoi(roim.getRoi(index))
                            IJ.saveAs(output_image, "PNG", output_image_path)
                        if table.size() != 0:
                            area = table.getValue("Area", index)
                            imp = IJ.openImage(image_path)            
                            IJ.run(imp, "Set Scale...", "distance=1 known=" +str(pixel_size)+ " pixel=1 unit=um global")
                            ImageConverter(imp).convertToGray8()
                            IJ.run(imp, "Invert", "") #because grey intesity is calculated as bright over dark, but in brightfield we calculate as dark over bright
                            #mean, median, int density, must be calculated from inverted image, std dev, mode etc, other parameters will not change, except skew and kurt positive becomes negative and vice versa. therefore you do not need to take some measurements from original image and some from inverted, take all from inverted
                            # Get the ROI from the ROI Manager
                            roim.select(index)
                            # Make sure the ROI is actively selected on the image
                            imp.setRoi(roim.getRoi(index))

                            if roi is not None:
                                imp.setRoi(roi)  # Set the ROI on the image
                                
                                # Get the dimensions of the original image
                                width, height = imp.getWidth(), imp.getHeight()

                                # Create a blank image with the same dimensions
                                blank_processor = ByteProcessor(width, height)
                                blank_processor.setColor(0)  # Set background to black
                                blank_processor.fill()  # Fill with black

                                # Draw the ROI onto the blank image
                                blank_processor.setColor(255)  # Set draw color to white
                                blank_processor.fill(roi)  # Fill the ROI with white

                                # Create an ImagePlus object from the blank_processor
                                mask_imp = ImagePlus("Mask", blank_processor)

                                # Ensure the mask is in binary format
                                IJ.run(mask_imp, "8-bit", "")  # Convert to 8-bit if necessary
                                IJ.setThreshold(mask_imp, 1, 255)  # Set threshold to make mask binary
                                IJ.run(mask_imp, "Convert to Mask", "")  # Convert to mask

                                # Save the mask as an image
                                # output_image_path = os.path.join(masks_subfolder_directory, filename.replace(".", "_mask"))
                                # IJ.saveAs(mask_imp, "PNG", output_image_path)
                                file_root, file_ext = os.path.splitext(filename)
                                output_image_path = os.path.join(masks_subfolder_directory, "{}_mask.png".format(file_root))
                                IJ.saveAs(mask_imp, "PNG", output_image_path)

                            rt = ResultsTable() #check this
                            # Prevent the Results Table window from opening after running the Measure command
                            #IJ.setOption("Show Results", False)
                            IJ.run(imp, "Measure", "")
                            # Retrieve the results table
                            
                            rt = ResultsTable.getResultsTable() #check this
                            # Loop over the rows in the Results Table (rt), it will be used to access values and write to output file, index for table has different rows because it measures more particles
                            for rt_index in range(rt.getCounter()):
                                unused_variable = 1
                            #print(table.size())
                            #print(rt.getCounter())
                            # Print contents of the table ResultsTable
                            #table.show("Table Results")
                            
                            # Invert ROI selection to obtain background regions
                            # Get the ROI from the ROI Manager
                            roim.select(index)
                            # Make sure the ROI is actively selected on the image
                            imp.setRoi(roim.getRoi(index))
                            IJ.run(imp, "Make Inverse", "")
                            # Calculate mean background intensity, fix this, threshold and remove black objects, then calculate mean of the remaining area, this will get rid of shadows and other debri that are not really background
                            stats_bg = imp.getStatistics()
                            background_mean_intensity = stats_bg.mean
                            # Calculate Corrected Total Fluorescence (CTF) for each particle
                            #ctf = (table.getValue("Area", index) * rt.getValue("Mean", rt_index)) - (table.getValue("Area", index) * background_mean_intensity)
                            ctf = rt.getValue("IntDen", rt_index) - (area * background_mean_intensity)
                            cmf = ctf / area #corrected mean fluorescence
                            # minumum and maximum intesity pixel corrected
                            # background_min_intensity = stats_bg.min
                            # corrected_min_intensity = (rt.getValue("Min", rt_index)) - background_min_intensity
                            # background_max_intensity = stats_bg.max
                            # corrected_max_intensity = (rt.getValue("Max", rt_index)) - background_max_intensity
                            corrected_min_intensity = (rt.getValue("Min", rt_index)) - background_mean_intensity
                            corrected_max_intensity = (rt.getValue("Max", rt_index)) - background_mean_intensity

                            
                            #centroid to center of mass
                            centers_distance_y = abs(rt.getValue("Y", rt_index) - rt.getValue("YM", rt_index))
                            centers_distance_x = abs(rt.getValue("X", rt_index) - rt.getValue("XM", rt_index))
                            centers_distance = ((centers_distance_x ** 2) + (centers_distance_y ** 2)) ** 0.5
                            
                            #feret coordinates FeretX and FeretY
                            
                            # normalize the feret angle according to the elipse angle, so we get a ratio independent of the x axis, maybe make it absolute, or normalize both to rectangle
                            feret_to_ellipse_angle =  abs(table.getValue("FeretAngle", index) - table.getValue("Angle", index))

                            # Store the area for the corresponding day-organoid combination
                            print("day " + str(day))
                            print("organoid " + str(organoid))
                            growth_rate = "NA"
                            growth_rate_pd = "NA"
                            if day != "NA": #only execute this part if day and organoid can be read from filename, otherwise mathematical operations will not work
                                if day not in day_organoid_area_map:
                                    day_organoid_area_map[day] = {}
                                    print(day_organoid_area_map)
                                day_organoid_area_map[day][organoid] = area
                                # print(day_organoid_area_map)                            
                                # Preprocess day keys to convert single-digit days to double-digit days
                                # day_organoid_area_map = {day.zfill(2): areas for day, areas in day_organoid_area_map.items()}
                                # day0_area_map = {day.zfill(2): areas for day, areas in day0_area_map.items()}
                                if day != "00":
                                    # Check if there are entries for day 0 in day_organoid_area_map
                                    if '00' in day_organoid_area_map and mean_area_day0 == 0:
                                        # Extract areas for day 0 organoids
                                        day0_organoid_areas = day_organoid_area_map['00']
                                        print("organoid " + str(organoid))
                                        print("day" + str(day))
                                        # Calculate area ratios relative to day 0 for each organoid on days other than day 0
                                        day0_area_map = day_organoid_area_map.get("00", {})
                                        mean_area_day0 = sum(day0_area_map.values()) / len(day0_area_map)  
                                        print("mean_area_day0", mean_area_day0)
                                    print("organoid " + str(organoid))
                                    print("growth rate calculated")
                                    day0_area = day0_area_map.get(organoid, mean_area_day0)  # Get the area for the organoid in day 0, defaulting to the mean value if not available
                                    growth_rate = area / day0_area
                                    #print("growth:" + str(growth_rate))
                                    #print(f"Day {day}, Organoid {organoid}: growth rate = {growth_rate}")
                                else:
                                    growth_rate = 1
                            diameter = 2 * math.sqrt(float(table.getValue("Area", index)) / (2 * math.pi))
                            criteria = table.getValue("Area", index) > area_threshold and table.getValue("Circ.", index) > circularity_threshold

                            output.write(str(subfolder) + ',' + filename + ',' + day + ',' + organoid + ',' + str(table.getValue("Area", index)) + ',' + str(growth_rate) + ',' +  str(table.getValue("Perim.", index)) + ',' + str(table.getValue("Feret", index)) + ',' + str(table.getValue("MinFeret", index)) + ',' + str(table.getValue("Major", index)) + ',' + str(table.getValue("Minor", index)) + ',' + str(table.getValue("AR", index)) + ',' + str(diameter) + ',' + str(table.getValue("Circ.", index)) + ',' + str(table.getValue("Round", index)) + ',' + str(table.getValue("Solidity", index)) + ',' + str(ctf) + ',' + str(cmf) + ',' + str(rt.getValue("IntDen", rt_index)) + ',' + str(rt.getValue("Mean", rt_index)) + ',' + str(rt.getValue("StdDev", rt_index)) + ',' + str(rt.getValue("Mode", rt_index)) + ',' + str(rt.getValue("Median", rt_index)) + ',' + str(corrected_min_intensity) + ',' + str(corrected_max_intensity) + ',' + str(rt.getValue("Skew", rt_index) * -1) + ',' + str(rt.getValue("Kurt", rt_index)) + ',' + str(centers_distance) + ',' + str(criteria))#  maybe I do not need to write as strings?

                            #we saved anything measured uneder minimum values wether or not meets criteria, so that we can inspect manually later, 
                            imp = IJ.openImage(image_path) 
                            # Ensure the image is in RGB format before drawing
                            #if not imp.getType() == ij.ImagePlus.COLOR_RGB:
                            IJ.run(imp, "RGB Color", "") #you can only draw color in rgb images, make sure images are rgb
                            # Get the ROI from the ROI Manager
                            #print("index for draw = " + str(index))
                            roim.select(index)
                            # Make sure the ROI is actively selected on the image
                            imp.setRoi(roim.getRoi(index))
                            # Set the drawing color, e.g., to red
                            IJ.setForegroundColor(255, 0, 255); # RGB values for magenta color
                            time.sleep(0.1)
                            # Draw the ROI onto the image
                            IJ.run(imp, "Line Width...", "line=11");
                            time.sleep(0.1)
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

# for sending to imagej console
# Close the log file
log_file.close()

# Reset stdout to original stdout
sys.stdout = original_stdout

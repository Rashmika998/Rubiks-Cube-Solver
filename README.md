# Rubiks-Cube-Solver
step 1<br></br>
Input images of Rubik's cube 
You will need to input two images of the Rubik's cube each showing three sides of the cube. You should do this as follows:
First place the cube showing three sides(side 1, side 2 and side 5) as illustrated in the image below.<br/>
<img src="https://user-images.githubusercontent.com/91874321/211251579-0021e8ad-5ca9-4430-aab6-6acc856f234e.png" width="500" height="400"/>

Then turn the cube 180 degrees so that the hidden sides(side 4, side 3 and side 6) are now facing front.
<img src="https://user-images.githubusercontent.com/91874321/211251639-3fc7dd4d-ef13-49e6-913b-d5cac26aef68.png" width="500" height="400"/>

step 2<br></br>
Save the correctly captured two images in the Input directory. 

step 3<br></br>
The 9 color pieces in each side are indexed in the order as shown in the images in step 1.
The program will then extract all squares that it can find, and will present you with a dialog to correct any faces it failed to identify.
In the dialog showed up, check whether all the extracted colors match up with the Rubik's cube adhering to the correct order of indexing.
If any squares have mistakes, double click on the square and press the first letter of the correct color (e.g. if the square should have been blue, type "b"). 
![image](https://user-images.githubusercontent.com/91874321/211254525-a7008fa5-b7ba-43f7-905f-60aa7fce4fc8.png)

You can press enter once the correct color has been entered, or just single-click anywhere to stop the color editor.
Once you have verified that all of the colors of the squares correctly match the input images, press q to exit the dialog.
Then the dialog showing the steps to solve the cube will be displayed.

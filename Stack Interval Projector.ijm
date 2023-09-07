slices=getNumber("prompt", 70) 
setBatchMode(true); //batch mode on
name=getTitle;
thickness=10
start_slice=7
for (i=0; i<slices; i+=1) {
selectImage(name); 
//run("Z Project...", "start="+(start_slice+i)+" stop="+(start_slice+i+thickness)+" projection=[Standard Deviation]");
run("Z Project...", "start="+(start_slice+i)+" stop="+(start_slice+i+thickness)+" projection=[Max Intensity]");
}

run("Images to Stack");
setBatchMode(false); //batch mode on
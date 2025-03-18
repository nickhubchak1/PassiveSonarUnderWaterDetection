# Under Water Passibe Acoustic Source Localization

We developed a model to track distances of the sea floor. 
The processed dataset is too large to store in github so we are storing it in a shared drive that can handle its compressed size (12.5 Gb) here: 

[SW96_CSDM.zip](https://drexel0-my.sharepoint.com/:u:/g/personal/nh585_drexel_edu/EetAsTAddv1NkG7vvt8LyzMBzpbld3eytqdwNIMFJz6APA?e=4pYVsr) 

Please download it and uncompress it and add it to the same folder as your **PassiveSonarUnderWaterDetection** cloned repo or forked branch lives locally. 


## Getting Started

First install the requirements needed for this project (NOTE: The SW96_CSDM.zip is needed as well please download and uncompross to local git repo)
Make sure to run from the **PassiveSonarUnderWaterDetection** folder: 
```bash

python install_requirements.py
```

Next run the flatten script to make the large dataset more managable: 

```bash
python flatten_sw96_CSDM.py
```

You should now see a flattened_data.h5 in your local repo folder. If not check terminal for errors and try again.

Now we are ready to train ! Go ahead and run the mlp.py (This is just a basic MLP with shallow layers)

```bash
python mlp.py
```


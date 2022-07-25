


% Transformations for augentation 


Val_Percent=20;

Val_Percent=Val_Percent/100;
original_images='./data_road/training/image_2';
original_masks='./data_road/training/gt_image_2';
original_ADI='./data_road/training/ADI';


target_images_training='./data_road_aug/train/images';
target_images_val='./data_road_aug/val/images';

target_masks_training='./data_road_aug/train/masks';
target_masks_val='./data_road_aug/val/masks';

target_ADI_training='./data_road_aug/train/ADI';
target_ADI_val='./data_road_aug/val/ADI';


% Rotation
rotation_angles=[5 , 10, 15, 17];

% translations
translations=[50, 100, 150];

% Scales (An auto plus 2)
scales=[1.1, 1.2, 1.3, 1.4, 1.5];

%Shear
shears=[10, 20 , 30 , 35, 40];



mkdir('./data_road_aug');
mkdir(target_images_training);
mkdir(target_masks_training);
mkdir(target_masks_val);


mkdir(target_ADI_training);
mkdir(target_ADI_val);
mkdir(target_images_val);



% Creating sorted lists of available files
images_list_uu=dir(strcat(original_images,'/uu_*'));
images_list_um=dir(strcat(original_images,'/um_*'));
images_list_umm=dir(strcat(original_images,'/umm_*'));


images_list_uu_mask=dir(strcat(original_masks,'/uu_*'));
images_list_um_mask=dir(strcat(original_masks,'/um_*'));
images_list_umm_mask=dir(strcat(original_masks,'/umm_*'));

count_uu=size(images_list_uu,1);
count_um=size(images_list_um,1);
count_umm=size(images_list_umm,1);

% Making train and valid
select_uu=zeros(1,count_uu);
select_um=zeros(1,count_um);
select_umm=zeros(1,count_umm);

rng('default')
rand_files_uu = randi([1 count_uu],floor(count_uu*Val_Percent),1)';
rand_files_um = randi([1 count_um],floor(count_um*Val_Percent),1)';
rand_files_umm = randi([1 count_umm],floor(count_umm*Val_Percent),1)';

select_uu(rand_files_uu)=1;
select_um(rand_files_um)=1;
select_umm(rand_files_umm)=1;

for i=1:count_uu
    % read image
    fname=strcat(images_list_uu(i).folder,'/',images_list_uu(i).name);
    img=imread(fname);   
    fname=strcat(original_masks,'/',images_list_uu_mask(i).name);
    mask=imread(fname);
    fname=strcat(original_ADI,'/',images_list_uu(i).name);   
    ADI=imread(fname);
    
    
    if (select_uu(i)==0)
        target_filename_image= fullfile(target_images_training,images_list_uu(i).name);        
        target_filename_mask= fullfile(target_masks_training,images_list_uu(i).name);
        target_filename_ADI=  fullfile(target_ADI_training,images_list_uu(i).name); 
        
        
        % Rotations Image     
        for j=1:size(rotation_angles,2)
            tform = randomAffine2d(Rotation=[-rotation_angles(j) rotation_angles( j)]); 
            I=img;
            target_filename= strcat(target_images_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);      
            
            I=mask;              
            target_filename= strcat(target_masks_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);   
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);
        end 
   
        
        % Translate images
        for j=1:size(translations,2)
            tform = randomAffine2d(XTranslation=[-translations(j) translations(j)],YTranslation=[-translations(j) translations(j)]); 

            I=img;
            target_filename= strcat(target_images_training,'/','trans_',num2str(translations(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename); 
            
            I=mask;
            target_filename= strcat(target_masks_training,'/','trans_',num2str(translations(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
   
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','trans_',num2str(translations(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);             
        end      
        % Rescale images                
        for j=1:size(scales,2)
            tform = randomAffine2d(Scale=[scales(j) scales(j)+0.1 ]);
            
            I=img;
            target_filename= strcat(target_images_training,'/','scale_',num2str(scales(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_training,'/','scale_',num2str(scales(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','scale_',num2str(scales(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
        end            
        
        %Image Shears
         for j=1:size(shears,2)
            tform = randomAffine2d(XShear=[-shears(j) shears(j)]);
            
            I=img;
            target_filename= strcat(target_images_training,'/','shears_',num2str(shears(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_training,'/','shears_',num2str(shears(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','shears_',num2str(shears(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
         end  
        
        % Reflections
        I=img;
        target_filename= strcat(target_images_training,'/','flip_','x','_',images_list_uu(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        I=mask;
        target_filename= strcat(target_masks_training,'/','flip_','x','_',images_list_uu(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        
        I=ADI;
        target_filename= strcat(target_ADI_training,'/','flip_','x','_',images_list_uu(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename);  
        
        %Color transformations
        % Satudation
        I=img;
        target_filename= strcat(target_images_training,'/','sat_','',images_list_uu(i).name);           
        X = jitterColorHSV(I,Saturation=[-0.4 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','sat_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','sat_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
        
        % Brightness
        I=img;
        target_filename= strcat(target_images_training,'/','bright_','',images_list_uu(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','bright_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','bright_','',images_list_uu(i).name);           
        imwrite(X,target_filename);
        
        % contrast jitter
        I=img;
        target_filename= strcat(target_images_training,'/','contrast_','',images_list_uu(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','contrast_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','contrast_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
        
    else
        target_filename_image= fullfile(target_images_val,images_list_uu(i).name);
        target_filename_mask= fullfile(target_masks_val,images_list_uu(i).name);
        target_filename_ADI=  fullfile(target_ADI_val,images_list_uu(i).name); 
        
        
        % Rotations Image     
        for j=1:size(rotation_angles,2)
            tform = randomAffine2d(Rotation=[-rotation_angles(j) rotation_angles( j)]); 
            I=img;
            target_filename= strcat(target_images_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);      
            
            I=mask;              
            target_filename= strcat(target_masks_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);   
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);
        end 
   
        
        % Translate images
        for j=1:size(translations,2)
            tform = randomAffine2d(XTranslation=[-translations(j) translations(j)],YTranslation=[-translations(j) translations(j)]); 

            I=img;
            target_filename= strcat(target_images_val,'/','trans_',num2str(translations(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename); 
            
            I=mask;
            target_filename= strcat(target_masks_val,'/','trans_',num2str(translations(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
   
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','trans_',num2str(translations(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);             
        end      
        % Rescale images                
        for j=1:size(scales,2)
            tform = randomAffine2d(Scale=[scales(j) scales(j)+0.1 ]);
            
            I=img;
            target_filename= strcat(target_images_val,'/','scale_',num2str(scales(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_val,'/','scale_',num2str(scales(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','scale_',num2str(scales(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
        end            
        %Image Shears
         for j=1:size(shears,2)
            tform = randomAffine2d(XShear=[-shears(j) shears(j)]);
            
            I=img;
            target_filename= strcat(target_images_val,'/','shears_',num2str(shears(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_val,'/','shears_',num2str(shears(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','shears_',num2str(shears(j)),'_',images_list_uu(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
         end          
        % Reflections
        I=img;
        target_filename= strcat(target_images_val,'/','flip_','x','_',images_list_uu(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        I=mask;
        target_filename= strcat(target_masks_val,'/','flip_','x','_',images_list_uu(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        
        I=ADI;
        target_filename= strcat(target_ADI_val,'/','flip_','x','_',images_list_uu(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
       
        %Color transformations
        % Satudation
        I=img;
        target_filename= strcat(target_images_val,'/','sat_','',images_list_uu(i).name);           
        X = jitterColorHSV(I,Saturation=[-0.4 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','sat_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','sat_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
        
        % Brightness
        I=img;
        target_filename= strcat(target_images_val,'/','bright_','',images_list_uu(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','bright_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','bright_','',images_list_uu(i).name);           
        imwrite(X,target_filename);
        
        % contrast jitter
        I=img;
        target_filename= strcat(target_images_val,'/','contrast_','',images_list_uu(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','contrast_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','contrast_','',images_list_uu(i).name);           
        imwrite(X,target_filename);  
        
    end
        imwrite(img,target_filename_image) ;  
        imwrite(mask,target_filename_mask);
        imwrite(ADI,target_filename_ADI);
        
        fprintf("Processing UU image: %d \n",i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:count_um
    % read image
    fname=strcat(images_list_um(i).folder,'/',images_list_um(i).name);
    img=imread(fname);   
    fname=strcat(original_masks,'/',images_list_um_mask(i).name);
    mask=imread(fname);
    fname=strcat(original_ADI,'/',images_list_um(i).name);   
    ADI=imread(fname);
    
    
    if (select_um(i)==0)
        target_filename_image= fullfile(target_images_training,images_list_um(i).name);        
        target_filename_mask= fullfile(target_masks_training,images_list_um(i).name);
        target_filename_ADI=  fullfile(target_ADI_training,images_list_um(i).name); 
        
        
        % Rotations Image     
        for j=1:size(rotation_angles,2)
            tform = randomAffine2d(Rotation=[-rotation_angles(j) rotation_angles( j)]); 
            I=img;
            target_filename= strcat(target_images_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);      
            
            I=mask;              
            target_filename= strcat(target_masks_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);   
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);
        end 
   
        
        % Translate images
        for j=1:size(translations,2)
            tform = randomAffine2d(XTranslation=[-translations(j) translations(j)],YTranslation=[-translations(j) translations(j)]); 

            I=img;
            target_filename= strcat(target_images_training,'/','trans_',num2str(translations(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename); 
            
            I=mask;
            target_filename= strcat(target_masks_training,'/','trans_',num2str(translations(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
   
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','trans_',num2str(translations(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);             
        end      
        % Rescale images                
        for j=1:size(scales,2)
            tform = randomAffine2d(Scale=[scales(j) scales(j)+0.1 ]);
            
            I=img;
            target_filename= strcat(target_images_training,'/','scale_',num2str(scales(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_training,'/','scale_',num2str(scales(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','scale_',num2str(scales(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
        end            
        
        %Image Shears
         for j=1:size(shears,2)
            tform = randomAffine2d(XShear=[-shears(j) shears(j)]);
            
            I=img;
            target_filename= strcat(target_images_training,'/','shears_',num2str(shears(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_training,'/','shears_',num2str(shears(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','shears_',num2str(shears(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
         end  
        
        % Reflections
        I=img;
        target_filename= strcat(target_images_training,'/','flip_','x','_',images_list_um(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        I=mask;
        target_filename= strcat(target_masks_training,'/','flip_','x','_',images_list_um(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        
        I=ADI;
        target_filename= strcat(target_ADI_training,'/','flip_','x','_',images_list_um(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename);  
        
        %Color transformations
        % Satudation
        I=img;
        target_filename= strcat(target_images_training,'/','sat_','',images_list_um(i).name);           
        X = jitterColorHSV(I,Saturation=[-0.4 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','sat_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','sat_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
        
        % Brightness
        I=img;
        target_filename= strcat(target_images_training,'/','bright_','',images_list_um(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','bright_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','bright_','',images_list_um(i).name);           
        imwrite(X,target_filename);
        
        % contrast jitter
        I=img;
        target_filename= strcat(target_images_training,'/','contrast_','',images_list_um(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','contrast_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','contrast_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
        
    else
        target_filename_image= fullfile(target_images_val,images_list_um(i).name);
        target_filename_mask= fullfile(target_masks_val,images_list_um(i).name);
        target_filename_ADI=  fullfile(target_ADI_val,images_list_um(i).name); 
        
        
        % Rotations Image     
        for j=1:size(rotation_angles,2)
            tform = randomAffine2d(Rotation=[-rotation_angles(j) rotation_angles( j)]); 
            I=img;
            target_filename= strcat(target_images_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);      
            
            I=mask;              
            target_filename= strcat(target_masks_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);   
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);
        end 
   
        
        % Translate images
        for j=1:size(translations,2)
            tform = randomAffine2d(XTranslation=[-translations(j) translations(j)],YTranslation=[-translations(j) translations(j)]); 

            I=img;
            target_filename= strcat(target_images_val,'/','trans_',num2str(translations(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename); 
            
            I=mask;
            target_filename= strcat(target_masks_val,'/','trans_',num2str(translations(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
   
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','trans_',num2str(translations(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);             
        end      
        % Rescale images                
        for j=1:size(scales,2)
            tform = randomAffine2d(Scale=[scales(j) scales(j)+0.1 ]);
            
            I=img;
            target_filename= strcat(target_images_val,'/','scale_',num2str(scales(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_val,'/','scale_',num2str(scales(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','scale_',num2str(scales(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
        end            
        %Image Shears
         for j=1:size(shears,2)
            tform = randomAffine2d(XShear=[-shears(j) shears(j)]);
            
            I=img;
            target_filename= strcat(target_images_val,'/','shears_',num2str(shears(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_val,'/','shears_',num2str(shears(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','shears_',num2str(shears(j)),'_',images_list_um(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
         end          
        % Reflections
        I=img;
        target_filename= strcat(target_images_val,'/','flip_','x','_',images_list_um(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        I=mask;
        target_filename= strcat(target_masks_val,'/','flip_','x','_',images_list_um(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        
        I=ADI;
        target_filename= strcat(target_ADI_val,'/','flip_','x','_',images_list_um(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
       
        %Color transformations
        % Satudation
        I=img;
        target_filename= strcat(target_images_val,'/','sat_','',images_list_um(i).name);           
        X = jitterColorHSV(I,Saturation=[-0.4 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','sat_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','sat_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
        
        % Brightness
        I=img;
        target_filename= strcat(target_images_val,'/','bright_','',images_list_um(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','bright_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','bright_','',images_list_um(i).name);           
        imwrite(X,target_filename);
        
        % contrast jitter
        I=img;
        target_filename= strcat(target_images_val,'/','contrast_','',images_list_um(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','contrast_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','contrast_','',images_list_um(i).name);           
        imwrite(X,target_filename);  
        
    end
        imwrite(img,target_filename_image) ;  
        imwrite(mask,target_filename_mask);
        imwrite(ADI,target_filename_ADI);
        
        fprintf("Processing um image: %d \n",i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:count_umm
    % read image
    fname=strcat(images_list_umm(i).folder,'/',images_list_umm(i).name);
    img=imread(fname);   
    fname=strcat(original_masks,'/',images_list_umm_mask(i).name);
    mask=imread(fname);
    fname=strcat(original_ADI,'/',images_list_umm(i).name);   
    ADI=imread(fname);
    
    
    if (select_umm(i)==0)
        target_filename_image= fullfile(target_images_training,images_list_umm(i).name);        
        target_filename_mask= fullfile(target_masks_training,images_list_umm(i).name);
        target_filename_ADI=  fullfile(target_ADI_training,images_list_umm(i).name); 
        
        
        % Rotations Image     
        for j=1:size(rotation_angles,2)
            tform = randomAffine2d(Rotation=[-rotation_angles(j) rotation_angles( j)]); 
            I=img;
            target_filename= strcat(target_images_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);      
            
            I=mask;              
            target_filename= strcat(target_masks_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);   
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','rot_',num2str(rotation_angles(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);
        end 
   
        
        % Translate images
        for j=1:size(translations,2)
            tform = randomAffine2d(XTranslation=[-translations(j) translations(j)],YTranslation=[-translations(j) translations(j)]); 

            I=img;
            target_filename= strcat(target_images_training,'/','trans_',num2str(translations(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename); 
            
            I=mask;
            target_filename= strcat(target_masks_training,'/','trans_',num2str(translations(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
   
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','trans_',num2str(translations(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);             
        end      
        % Rescale images                
        for j=1:size(scales,2)
            tform = randomAffine2d(Scale=[scales(j) scales(j)+0.1 ]);
            
            I=img;
            target_filename= strcat(target_images_training,'/','scale_',num2str(scales(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_training,'/','scale_',num2str(scales(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','scale_',num2str(scales(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
        end            
        
        %Image Shears
         for j=1:size(shears,2)
            tform = randomAffine2d(XShear=[-shears(j) shears(j)]);
            
            I=img;
            target_filename= strcat(target_images_training,'/','shears_',num2str(shears(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_training,'/','shears_',num2str(shears(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_training,'/','shears_',num2str(shears(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
         end  
        
        % Reflections
        I=img;
        target_filename= strcat(target_images_training,'/','flip_','x','_',images_list_umm(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        I=mask;
        target_filename= strcat(target_masks_training,'/','flip_','x','_',images_list_umm(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        
        I=ADI;
        target_filename= strcat(target_ADI_training,'/','flip_','x','_',images_list_umm(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename);  
        
        %Color transformations
        % Satudation
        I=img;
        target_filename= strcat(target_images_training,'/','sat_','',images_list_umm(i).name);           
        X = jitterColorHSV(I,Saturation=[-0.4 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','sat_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','sat_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
        
        % Brightness
        I=img;
        target_filename= strcat(target_images_training,'/','bright_','',images_list_umm(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','bright_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','bright_','',images_list_umm(i).name);           
        imwrite(X,target_filename);
        
        % contrast jitter
        I=img;
        target_filename= strcat(target_images_training,'/','contrast_','',images_list_umm(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_training,'/','contrast_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_training,'/','contrast_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
        
    else
        target_filename_image= fullfile(target_images_val,images_list_umm(i).name);
        target_filename_mask= fullfile(target_masks_val,images_list_umm(i).name);
        target_filename_ADI=  fullfile(target_ADI_val,images_list_umm(i).name); 
        
        
        % Rotations Image     
        for j=1:size(rotation_angles,2)
            tform = randomAffine2d(Rotation=[-rotation_angles(j) rotation_angles( j)]); 
            I=img;
            target_filename= strcat(target_images_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);      
            
            I=mask;              
            target_filename= strcat(target_masks_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);   
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','rot_',num2str(rotation_angles(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);
        end 
   
        
        % Translate images
        for j=1:size(translations,2)
            tform = randomAffine2d(XTranslation=[-translations(j) translations(j)],YTranslation=[-translations(j) translations(j)]); 

            I=img;
            target_filename= strcat(target_images_val,'/','trans_',num2str(translations(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename); 
            
            I=mask;
            target_filename= strcat(target_masks_val,'/','trans_',num2str(translations(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
   
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','trans_',num2str(translations(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);             
        end      
        % Rescale images                
        for j=1:size(scales,2)
            tform = randomAffine2d(Scale=[scales(j) scales(j)+0.1 ]);
            
            I=img;
            target_filename= strcat(target_images_val,'/','scale_',num2str(scales(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_val,'/','scale_',num2str(scales(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','scale_',num2str(scales(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
        end            
        %Image Shears
         for j=1:size(shears,2)
            tform = randomAffine2d(XShear=[-shears(j) shears(j)]);
            
            I=img;
            target_filename= strcat(target_images_val,'/','shears_',num2str(shears(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);    
    
            I=mask;
            target_filename= strcat(target_masks_val,'/','shears_',num2str(shears(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);       
            
            I=ADI;
            target_filename= strcat(target_ADI_val,'/','shears_',num2str(shears(j)),'_',images_list_umm(i).name);   
            outputView = affineOutputView(size(I),tform);
            X = imwarp(I,tform,OutputView=outputView);  
            imwrite(X,target_filename);        
         end          
        % Reflections
        I=img;
        target_filename= strcat(target_images_val,'/','flip_','x','_',images_list_umm(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        I=mask;
        target_filename= strcat(target_masks_val,'/','flip_','x','_',images_list_umm(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
        
        I=ADI;
        target_filename= strcat(target_ADI_val,'/','flip_','x','_',images_list_umm(i).name);   
        X=flipdim(I,2);
        imwrite(X,target_filename); 
        
       
        %Color transformations
        % Satudation
        I=img;
        target_filename= strcat(target_images_val,'/','sat_','',images_list_umm(i).name);           
        X = jitterColorHSV(I,Saturation=[-0.4 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','sat_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','sat_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
        
        % Brightness
        I=img;
        target_filename= strcat(target_images_val,'/','bright_','',images_list_umm(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','bright_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','bright_','',images_list_umm(i).name);           
        imwrite(X,target_filename);
        
        % contrast jitter
        I=img;
        target_filename= strcat(target_images_val,'/','contrast_','',images_list_umm(i).name);           
        X = jitterColorHSV(I,Brightness=[-0.3 -0.1]); 
        imwrite(X,target_filename);  

        X=mask;
        target_filename= strcat(target_masks_val,'/','contrast_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
    
        X=ADI;
        target_filename= strcat(target_ADI_val,'/','contrast_','',images_list_umm(i).name);           
        imwrite(X,target_filename);  
        
    end
        imwrite(img,target_filename_image) ;  
        imwrite(mask,target_filename_mask);
        imwrite(ADI,target_filename_ADI);
        
        fprintf("Processing umm image: %d \n",i);
end



# Original Author: Alexi Musick @ Alexi.musick@sjsu.edu

# Import packages
import os
import configparser
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import photutils as utils
from regions import Regions
from tqdm import tqdm
from matplotlib.colors import LogNorm
from sklearn.linear_model import LinearRegression

# https://www.astropy.org/
from astropy.io import fits
from astropy.convolution import convolve
from astropy.table import Table
from astropy.visualization import SqrtStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import match_coordinates_sky, SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

# https://photutils.readthedocs.io/en/stable/
from photutils.utils import calc_total_error
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.aperture import CircularAperture
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources, SourceFinder, SourceCatalog

# WCS data
filename = get_pkg_data_filename('HSC_images\PERSEUS_HSC-G_img.fits')
hdulist = fits.open(filename)
hdu = hdulist[0]
wcs = WCS(hdu.header)


def process(data, x_start, y_start):
    # Background subtraction
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data -= bkg.background

    # Source detection and deblending
    print('Detecting sources...')
    threshold = 1.5 * bkg.background_rms
    kernel = make_2dgaussian_kernel(2.5, size=5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    segment_map = detect_sources(convolved_data, threshold, npixels=10)
    segm_deblend = deblend_sources(convolved_data, segment_map, npixels=10, nlevels=32, contrast=0.001,
                                   progress_bar=False)

    # General Image Segmentation View
    cmap2 = segm_deblend.cmap
    norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 27.0))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Background-subtracted Data')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
    ax2.set_title('Segmentation Image')
    ax3.imshow(segm_deblend, origin='lower', cmap=cmap2, interpolation='nearest')
    ax3.set_title('Deblended Segments')

    # Beginning of Cataloging
    print('Cataloging...')
    cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data, wcs=wcs)
    print(cat)
    tbl = cat.to_table()

    # Save the tbl object to a CSV file
    output_filename = 'table_data.csv'
    tbl.write(output_filename, format='csv', overwrite=True)

    user_input = input('Print table? (y/n):')
    if user_input == 'y':
        tbl.pprint(max_width=500)
    else:
        print(f'Table saved to {output_filename}.')

    # Save region file in the "regions" folder
    print('Creating regions file...')
    folder_name = 'regions'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = os.path.join(folder_name, 'my_region_file.reg')
    print(f'File saved as {filename}.')
    with open(filename, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write('fk5\n')  # Set the coordinate system to FK5 (RA, Dec) in degrees

    with open(filename, 'a') as f:
        xcentroids = tbl['xcentroid'].data + x_start
        ycentroids = tbl['ycentroid'].data + y_start

        for i, (x, y) in enumerate(zip(xcentroids, ycentroids)):
            source_id = f'PCHSC-{i + 1:04}'
            coord = wcs.pixel_to_world(x, y)
            ra = coord.ra.deg
            dec = coord.dec.deg
            line = f'circle({ra:.6f},{dec:.6f},5.0") # text = {{ {source_id} }}\n'
            f.write(line)

    print('Matching region files...')
    # Match coordinates in new catalog with Wittmann catalog
    pcc = Regions.read('regions/PCC_cat.reg', format='ds9')
    pchsc = Regions.read('regions/my_region_file.reg', format='ds9')

    pcc_coords = []
    pchsc_coords = []
    pcc_ids = []
    pchsc_ids = []

    for pcc_region in pcc:
        source_id = pcc_region.meta.get('text')
        ra, dec = pcc_region.center.ra.deg, pcc_region.center.dec.deg
        pcc_coords.append((ra, dec))
        pcc_ids.append(source_id)

    for pchsc_region in pchsc:
        source_id = pchsc_region.meta.get('text')
        ra, dec = pchsc_region.center.ra.deg, pchsc_region.center.dec.deg
        pchsc_coords.append((ra, dec))
        pchsc_ids.append(source_id)

    # Convert the coordinate lists to SkyCoord objects
    pcc_coords = SkyCoord(ra=[coord[0] for coord in pcc_coords] * u.deg, dec=[coord[1] for coord in pcc_coords] * u.deg)
    pchsc_coords = SkyCoord(ra=[coord[0] for coord in pchsc_coords] * u.deg,
                            dec=[coord[1] for coord in pchsc_coords] * u.deg)

    # Match coordinates using match_coordinates_sky
    idx, sep2d, dist3d = match_coordinates_sky(pcc_coords, pchsc_coords)

    # Create a dictionary to store the matching results
    matching_sources = {}
    for pcc_index, pchsc_index in enumerate(idx):
        if sep2d[pcc_index] < 0.001 * u.deg:  # Threshold
            source_id_pcc = pcc_ids[pcc_index]
            source_id_pchsc = pchsc_ids[pchsc_index]
            matching_sources[source_id_pcc] = source_id_pchsc

    df_matched_sources = pd.DataFrame.from_dict(matching_sources, orient='index', columns=['pchsc'])
    df_matched_sources['pcc'] = df_matched_sources.index
    df_matched_sources.reset_index(drop=True, inplace=True)
    output_filename = 'matched_sources.xlsx'
    df_matched_sources.to_excel(output_filename, index=False, engine='openpyxl')

    df_mflag = pd.read_excel('mflag_PCC_cat_full.xlsx')
    mflag_tuples_array = np.array(list(zip(df_mflag['pcc'], df_mflag['mflag'])))
    df_match = pd.read_excel('matched_sources.xlsx')
    match_tuples_array = np.array(list(zip(df_match['pchsc'], df_match['pcc'])))
    match_tuples_array = [(pchsc.strip(), pcc.strip()) for pchsc, pcc in match_tuples_array]

    mapping_dict = {item[0]: item[1] for item in mflag_tuples_array}
    updated_rows = []

    for row in match_tuples_array:
        pchsc, pcc = row

        # Check if the pcc exists in the mapping_dict
        if pcc in mapping_dict:
            mflag = mapping_dict[pcc]
            updated_row = (pchsc, pcc, mflag)  # Tuple of 'pchsc', 'pcc', and 'mflag'
        else:
            updated_row = (pchsc, pcc, None)  # Tuple with 'mflag' set to None if not found in mapping_dict

        updated_rows.append(updated_row)

    # Create a DataFrame from the updated_rows list
    df_updated = pd.DataFrame(updated_rows, columns=['pchsc', 'pcc', 'mflag'])
    df_updated['mflag'] = pd.to_numeric(df_updated['mflag'])
    df_updated.to_excel('matched_sources_mflag.xlsx', index=False)
    print('Region files matched and added morphology flags...')

    # Plot apertures
    print('Plotting apertures...')
    norm = simple_norm(data, 'sqrt')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 18.0))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Data')
    ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap, interpolation='nearest')
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax1, color='white', lw=1.2)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=1.2)

    # Source Identification
    print('Identifying stellar sources...')
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    iraffind = IRAFStarFinder(fwhm=2.5, threshold=3. * std, sharplo=0.0, sharphi=6.0, roundlo=0.0, roundhi=2.5,
                              sigma_radius=10)
    sources = iraffind(data - median)
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f'

    # Plot detected sources
    print('Plotting sources...')
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)
    plt.imshow(data, cmap='viridis', origin='lower', norm=LogNorm(), interpolation='nearest')
    plt.colorbar()
    apertures.plot(color='magenta', lw=1.5, alpha=0.5)

    # Save data to a csv file
    output_filename = 'stellar_output.csv'
    sources.write(output_filename, format='csv', overwrite=True)
    user_input = input('Print table? (y/n):')
    if user_input == 'y':
        sources.pprint(max_width=100)
    else:
        print(f'Table saved to {output_filename}.')

    print('Creating regions file...')
    folder_name = 'regions'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = os.path.join(folder_name, 'stellar_region.reg')
    print(f'File saved as {filename}.')
    with open(filename, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write('fk5\n')  # Set the coordinate system to FK5 (RA, Dec) in degrees
    with open(filename, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write('fk5\n')  # Set the coordinate system to FK5 (RA, Dec) in degrees

    with open(filename, 'a') as f:
        xcentroids = sources['xcentroid'].data + x_start
        ycentroids = sources['ycentroid'].data + y_start

        for i, (x, y) in enumerate(zip(xcentroids, ycentroids)):
            stellar_id = f'STELLAR-{i + 1:04}'
            coord = wcs.pixel_to_world(x, y)
            ra = coord.ra.deg
            dec = coord.dec.deg
            line = f'circle({ra:.6f},{dec:.6f},5.0") # text = {{ {stellar_id} }}\n'
            f.write(line)

    print('Matching stellar region files...')
    # Match coordinates in new catalog with Wittmann catalog
    stellar = Regions.read('regions/stellar_region.reg', format='ds9')
    pchsc = Regions.read('regions/my_region_file.reg', format='ds9')

    stellar_coords = []
    pchsc_coords = []
    stellar_ids = []
    pchsc_ids = []

    for stellar_region in stellar:
        source_id = stellar_region.meta.get('text')
        ra, dec = stellar_region.center.ra.deg, stellar_region.center.dec.deg
        stellar_coords.append((ra, dec))
        stellar_ids.append(source_id)

    for pchsc_region in pchsc:
        source_id = pchsc_region.meta.get('text')
        ra, dec = pchsc_region.center.ra.deg, pchsc_region.center.dec.deg
        pchsc_coords.append((ra, dec))
        pchsc_ids.append(source_id)

    # Convert the coordinate lists to SkyCoord objects
    stellar_coords = SkyCoord(ra=[coord[0] for coord in stellar_coords] * u.deg,
                              dec=[coord[1] for coord in stellar_coords] * u.deg)
    pchsc_coords = SkyCoord(ra=[coord[0] for coord in pchsc_coords] * u.deg,
                            dec=[coord[1] for coord in pchsc_coords] * u.deg)

    # Match coordinates using match_coordinates_sky
    idx, sep2d, dist3d = match_coordinates_sky(stellar_coords, pchsc_coords)

    # Create a dictionary to store the matching results
    matching_sources = {}
    for stellar_index, pchsc_index in enumerate(idx):
        if sep2d[stellar_index] < 0.001 * u.deg:  # Threshold
            source_id_stellar = stellar_ids[stellar_index]
            source_id_pchsc = pchsc_ids[pchsc_index]
            matching_sources[source_id_stellar] = source_id_pchsc

    df_matched_sources_stellar = pd.DataFrame.from_dict(matching_sources, orient='index', columns=['pchsc'])
    df_matched_sources_stellar['stellar'] = df_matched_sources_stellar.index
    df_matched_sources_stellar.reset_index(drop=True, inplace=True)
    output_filename = 'matched_sources_stellar.xlsx'
    df_matched_sources_stellar.to_excel(output_filename, index=False, engine='openpyxl')

    df_matched_sources_mflag = pd.read_excel('matched_sources_mflag.xlsx')
    matched_sources_tuples_array = np.array(list(
        zip(df_matched_sources_mflag['pchsc'], df_matched_sources_mflag['pcc'], df_matched_sources_mflag['mflag'])))
    df_stellar = pd.read_excel('matched_sources_stellar.xlsx')
    matched_stellar_tuples_array = np.array(list(zip(df_stellar['pchsc'], df_stellar['stellar'])))
    matched_stellar_tuples_array = [(pchsc.strip(), stellar.strip()) for pchsc, stellar in matched_stellar_tuples_array]

    mapping_dict = {item[0]: item[1] for item in matched_stellar_tuples_array}
    updated_rows = []

    for row in matched_sources_tuples_array:
        pchsc, pcc, mflag = row

        # Check if the pchsc exists in the mapping_dict
        if pchsc in mapping_dict:
            stellar = mapping_dict[pchsc]
            updated_row = (pchsc, pcc, mflag, stellar)  # Tuple of 'pchsc', 'pcc', and 'mflag' and 'stellar'
        else:
            updated_row = (pchsc, pcc, mflag, None)  # Tuple with 'stellar' set to None if not found in mapping_dict

        updated_rows.append(updated_row)

    # Create a DataFrame from the updated_rows list
    df_updated = pd.DataFrame(updated_rows, columns=['pchsc', 'pcc', 'mflag', 'stellar'])
    df_updated['mflag'] = pd.to_numeric(df_updated['mflag'])
    df_updated.to_excel('matched_sources_mflag_stellar.xlsx', index=False)
    print('Region files matched and added stellar ID...')

    # CLean up excel sheet
    df = pd.read_excel('matched_sources_mflag_stellar.xlsx')

    # Remove prefixes from the specified columns
    columns_to_remove_prefix = ['pchsc', 'pcc', 'stellar']  # Replace with the actual column names in your Excel file
    prefixes_to_remove = ['PCHSC-', 'PCC-', 'STELLAR-']

    for col_name in columns_to_remove_prefix:
        for prefix in prefixes_to_remove:
            df[col_name] = df[col_name].str.replace(prefix, '')

    # Save the modified DataFrame to a new Excel file
    df['pchsc'] = pd.to_numeric(df['pchsc'])
    df['pcc'] = pd.to_numeric(df['pcc'])
    df['stellar'] = pd.to_numeric(df['stellar'])
    output_excel_file_path = 'matched_sources_mflag_stellar.xlsx'
    df.to_excel(output_excel_file_path, index=False)

    # Plot FWHM vs. Magnitude / Sharpness vs. Roundness for stars
    sourceid, fwhm, magnitude, sharpness, roundness = sources['id'], sources['fwhm'], sources['mag'], sources[
        'sharpness'], sources['roundness']

    # Read the Excel file
    df = pd.read_excel('matched_sources_mflag_stellar.xlsx')

    # Create a list of tuples containing (matching stellar value, corresponding mflag value)
    matching_stellar_mflag = []
    non_matching_objects = []

    for sourceid_value in sourceid:
        matching_row = df[df['stellar'] == sourceid_value]
        if not matching_row.empty:
            matching_stellar_value = matching_row.iloc[0]['stellar']
            corresponding_mflag = matching_row.iloc[0]['mflag']
            matching_stellar_mflag.append((matching_stellar_value, corresponding_mflag))
        else:
            non_matching_objects.append(sourceid_value)

    galaxy_mask = np.isin(sourceid, [tup[0] for tup in matching_stellar_mflag])
    objects_mask = np.isin(sourceid, non_matching_objects)

    average_fwhm = np.mean(fwhm)

    print('Plotting matched sources with morphology...')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Scatter plot other points
    ax1.scatter(magnitude[~galaxy_mask], fwhm[~galaxy_mask], marker='.', color='lightgrey')
    ax1.scatter(magnitude[~objects_mask], fwhm[~objects_mask], marker='.', color='lightgrey')
    ax1.scatter(magnitude[objects_mask], fwhm[objects_mask], marker='.', color='lightgrey')

    ax1.set_xlabel('Magnitude (mag)')
    ax1.set_ylabel('FWHM')
    ax1.set_title('Magnitude vs. FWHM')

    ax1.axhline(average_fwhm, color='darkgrey', linestyle='--', label='Average FWHM')
    model = LinearRegression()
    model.fit(magnitude[:, np.newaxis], fwhm)
    fwhm_predicted = model.predict(magnitude[:, np.newaxis])
    ax1.plot(magnitude, fwhm_predicted, color='darkgrey', label='Trend Line')

    mflag_title = {
        2: 'Likely BG ETG/ UR SS',
        3: 'Cluster or BG galaxy w/ LTG',
        4: 'Cluster or BG galaxy w/ weak SS',
        5: 'Likely Cluster or BG EDG',
        6: 'Likely MS in BG',
        7: 'ES',
    }

    # Create empty scatter plots for the legend
    for mflag in mflag_title:
        color = cm.tab10(mflag - 1)  # Subtract 1 to align with colormap indices
        ax1.scatter([], [], marker='o', color=color, label=mflag_title[mflag])

    # Scatter plot galaxies with different colors based on Mflag
    for i, (source_id, mflag) in enumerate(matching_stellar_mflag):
        color = cm.tab10(mflag - 1)  # Subtract 1 to align with colormap indices
        ax1.scatter(magnitude[i], fwhm[i], marker='o', color=color)

    # Display only the legend without titles on the scatter plot
    ax1.legend(fontsize='small', loc='lower right')

    # Scatter plot other points on ax2
    ax2.scatter(roundness[~galaxy_mask], sharpness[~galaxy_mask], marker='.', color='lightgrey')
    ax2.scatter(roundness[~objects_mask], sharpness[~objects_mask], marker='.', color='lightgrey')
    ax2.scatter(roundness[objects_mask], sharpness[objects_mask], marker='.', color='lightgrey')

    # Create empty scatter plots for the legend
    for mflag in mflag_title:
        color = cm.tab10(mflag - 1)  # Subtract 1 to align with colormap indices
        ax2.scatter([], [], marker='o', color=color, label=mflag_title[mflag])

    # Scatter plot galaxies with different colors based on Mflag on ax2
    for i, (source_id, mflag) in enumerate(matching_stellar_mflag):
        color = cm.tab10(mflag - 1)  # Subtract 1 to align with colormap indices
        ax2.scatter(roundness[i], sharpness[i], marker='o', color=color)

    # Display only the legend without titles on the scatter plot on ax2
    ax2.legend(fontsize='small', loc='upper right')

    # Scatter plot other points on ax3
    ax3.scatter(magnitude[~galaxy_mask], sharpness[~galaxy_mask], marker='.', color='lightgrey')
    ax3.scatter(magnitude[~objects_mask], sharpness[~objects_mask], marker='.', color='lightgrey')
    ax3.scatter(magnitude[objects_mask], sharpness[objects_mask], marker='.', color='lightgrey')

    # Create empty scatter plots for the legend
    for mflag in mflag_title:
        color = cm.tab10(mflag - 1)  # Subtract 1 to align with colormap indices
        ax3.scatter([], [], marker='o', color=color, label=mflag_title[mflag])

    # Scatter plot galaxies with different colors based on Mflag on ax3
    for i, (source_id, mflag) in enumerate(matching_stellar_mflag):
        color = cm.tab10(mflag - 1)  # Subtract 1 to align with colormap indices
        ax3.scatter(magnitude[i], sharpness[i], marker='o', color=color)

    # Display only the legend without titles on the scatter plot on ax3
    ax3.legend(fontsize='small', loc='upper left')

    plt.tight_layout()
    plt.rcParams.update({'font.size': 12})
    plt.show()

    print(f'The average FWHM value for the set of stars is: {average_fwhm}.')

    config_file = 'data_limits.ini'
    config = configparser.ConfigParser()

    def get_file_paths(directory):
        file_paths = []
        bands = []

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_paths.append(filepath)
                bands.append(get_band_from_filename(filename))

        return file_paths, bands

    def get_band_from_filename(filename):
        return filename.split('-')[1].split('_')[0] + '_band'

    def process_files(file_paths):
        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
            hdu_list = fits.open(file_path)
            hdu_list.info()
            data = hdu_list[0].data

            # Check if the configuration file exists and if user wants to override
            if os.path.exists(config_file):
                override = input(
                    f"The configuration file '{config_file}' already exists.\nDo you want to override it with new limits? (y/n): ")
                if override.lower() == 'y':
                    x_start = int(input("Enter the starting x limit: "))
                    x_end = int(input("Enter the ending x limit: "))
                    y_start = int(input("Enter the starting y limit: "))
                    y_end = int(input("Enter the ending y limit: "))
                    config['DataLimits'] = {
                        'x_start': str(x_start),
                        'x_end': str(x_end),
                        'y_start': str(y_start),
                        'y_end': str(y_end)
                    }
                    with open(config_file, 'w') as configfile:
                        config.write(configfile)
                else:
                    # Read the existing limits from the configuration file
                    config.read(config_file)
                    x_start = int(config['DataLimits']['x_start'])
                    x_end = int(config['DataLimits']['x_end'])
                    y_start = int(config['DataLimits']['y_start'])
                    y_end = int(config['DataLimits']['y_end'])
            else:
                # Configuration file does not exist, ask the user for limits
                x_start = int(input("Enter the starting x limit: "))
                x_end = int(input("Enter the ending x limit: "))
                y_start = int(input("Enter the starting y limit: "))
                y_end = int(input("Enter the ending y limit: "))
                config['DataLimits'] = {
                    'x_start': str(x_start),
                    'x_end': str(x_end),
                    'y_start': str(y_start),
                    'y_end': str(y_end)
                }
                with open(config_file, 'w') as configfile:
                    config.write(configfile)

            # Slice the data based on x and y limits
            data = data[y_start:y_end, x_start:x_end]

            print('Processing data...')
            process(data, x_start, y_start)

    def main():
        directory = 'HSC_images'  # Replace with the path to your directory

        file_paths, bands = get_file_paths(directory)

        print('Available bands in folder:')
        print('\n'.join(bands))

        process_files(file_paths)

    if __name__ == '__main__':
        main()
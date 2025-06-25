from setuptools import find_packages, setup

package_name = 'storagy_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='storagy',
    maintainer_email='wjddn2006@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmd_vel_example = storagy_drive.cmd_vel_example:main',
            'scan_example = storagy_drive.scan_example:main',
            'image_example = storagy_drive.image_example:main',
            'cmd_test = storagy_drive.cmd_test:main',
            'img_test = storagy_drive.img_test:main',
            'storagy_drive = storagy_drive.storagy_drive:main',


        ],
    },
)

package_path = pwd;
package_path = strcat (package_path, '/cns');
addpath (package_path)

package_path = pwd;
package_path = strcat (package_path, '/cnpkg');
addpath (package_path)

package_path = pwd;
package_path = strcat (package_path, '/cnpkg3');
addpath (package_path)

cd cns
cns_path
cd ..

clear package_path


EMroot = '/home/sturaga/EM';
addpath(EMroot);
EMstartup

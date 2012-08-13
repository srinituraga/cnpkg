% package_path = pwd;
% package_path = strcat (package_path, '/cns');
% addpath (package_path)
% 
% package_path = pwd;
% package_path = strcat (package_path, '/cnpkg');
% addpath (package_path)
% 
% package_path = pwd;
% package_path = strcat (package_path, '/cnpkg3');
% addpath (package_path)
% 
% cd cns
% cns_path
% cd ..
% 
% clear package_path

if ~isdeployed,

package_path = pwd;
package_path = strcat (package_path, '/cns2');
addpath (package_path)

package_path = pwd;
package_path = strcat (package_path, '/cnpkg');
addpath (package_path)

package_path = pwd;
package_path = strcat (package_path, '/cnpkg4');
addpath (package_path)

package_path = pwd;
package_path = strcat (package_path, '/cnpkg4/DAGsurgery/bin');
addpath (package_path)
javaaddpath ([package_path '/DAGedit.jar'])

cd cns2
cns_path
cd ..

clear package_path

EMroot = '/home/sturaga/EM';
addpath(EMroot);
EMstartup

end

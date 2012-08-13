cns done;clear all;close all;clc

m{1}        = denoise_mknet;
m_trained{1} = cnpkg_train_randindex(m{1},true);
m_trained{1} = cns('update',m_trained{1});
% cns('done')

% m_tested  = cnpkg_test(m_trained{1},true);
% keyboard
for hl = 2:6
    m{hl} = cnpkg_mknet_addlayer(m_trained{hl-1});
    m_trained{hl} = cnpkg_train_randindex(m{hl},true);
    m_trained{hl} = cns('update',m_trained{hl});
%     cns('done')
    
% m_tested  = cnpkg_test(m_trained{hl},true);
% keyboard
    
end


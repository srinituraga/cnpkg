function username = get_username()
% gets the username from unix()

[aa,username]=unix('whoami');
% sometimes (as on brainiac), the last character is a \n. get rid of all non-letters.
username = username(isletter(username));

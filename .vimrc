"execute pathogen#infect()

:syntax on
:set ts=4
:set nu
:set vb
:set expandtab ts=4 sw=4 ai
:autocmd InsertEnter,InsertLeave * set cul!
:match ErrorMsg '\%>80v.\+\|\s\+$'

"set statusline+=%#warningmsg#
"set statusline+=%{SyntasticStatuslineFlag()}
"set statusline+=%*

"let g:syntastic_always_populate_loc_list = 1
"let g:syntastic_auto_loc_list = 1
"let g:syntastic_check_on_open = 1
"let g:syntastic_check_on_wq = 0

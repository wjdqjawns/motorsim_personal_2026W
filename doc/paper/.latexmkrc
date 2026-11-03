# --- output directory ---
$out_dir = 'build';
$aux_dir = 'build';

# --- emulate aux separation ---
$emulate_aux = 1;

# --- output PDF name ---
$jobname = 'motor_simulation';

# --- compiler ---
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# --- bibliography ---

$bibtex = 'bibtex %O %S';

# --- copy PDF to root for convenience ---
use File::Copy;
END {
    if (-e "build/$jobname.pdf") {
        copy("build/$jobname.pdf", "$jobname.pdf");
    }
}

# create build folder
if (!-d 'build') { mkdir 'build'; }
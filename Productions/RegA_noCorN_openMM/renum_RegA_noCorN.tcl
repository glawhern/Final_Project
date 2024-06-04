# Get the current working directory
set current_dir [pwd]

# Load the PDB file
mol new [file join $current_dir "RegA_noCorN.pdb"]

# Load the XTC file with a stride of 5
mol addfile [file join $current_dir "RegA_noCorN.xtc"] step 5

# Renumber residues
set sel [atomselect top "all"]
set resids [$sel get resid]

# Adjust residue numbers
set new_resids {}
foreach resid $resids {
    lappend new_resids [expr {$resid + 21}]
}
$sel set resid $new_resids

# Update the selection to reflect the changes
$sel update

# Save the renumbered PDB file
set new_pdb_file [file join $current_dir "renum_RegA_noCorN.pdb"]
$sel writepdb $new_pdb_file

# Save the renumbered XTC file
set new_xtc_file [file join $current_dir "renum_RegA_noCorN.xtc"]
animate write xtc $new_xtc_file beg 0 end -1 sel [atomselect top "all"]

# Close the current molecule
mol delete top

# Load the renumbered PDB and XTC files
mol new $new_pdb_file
mol addfile $new_xtc_file

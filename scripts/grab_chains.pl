#!/usr/bin/perl

# Brian Pierce
# 5/31/2005
# grab the specified chains from the input file, in the order
# that they are specified

use strict;

my $pdb_file = $ARGV[0];
my $first_chain = $ARGV[1];

if (($pdb_file eq "") || ($first_chain eq "")) { die("usage: grab_chains.pl pdb_file chn1 [chn2] [chn3]\n"); }

open(PDB, $pdb_file) || die("unable to open the pdb file: $pdb_file\n");
my @pdb_lines = <PDB>;
close PDB;

# go through the block file, block the appropriate atoms
for (my $i = 1; $i < @ARGV; $i++)
{
    my $chain = $ARGV[$i];
    if ($chain eq "_") { $chain = " "; } # special whitespace character
    my $num_found = 0;
    foreach my $stuff (@pdb_lines)
    {
	if ((substr($stuff, 0, 4) eq "ATOM") || (substr($stuff, 0, 6) eq "HETATM"))
	{
	    my $ch = substr($stuff, 21, 1);
	    if ($ch eq $chain)
	    {
		print $stuff;
		$num_found++;
	    }
	}
	
    }
    if ($num_found == 0) { print "error: unable to find chain $chain\n"; }
}


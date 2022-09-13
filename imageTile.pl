#!/usr/bin/perl --
use strict;
use warnings;
use utf8;

use Imager;
use feature qw(say);

my $xRepeat=3;
my $yRepeat=3;

@ARGV or die "usage: $0 (infile.png)\n";

for my $inFile (@ARGV){
	my $outFile = $inFile;

	$outFile =~ s/\.png$/\.tile.jpg/i
		or die "not .png ! $inFile\n";

	if( -e $outFile ){
		say "skip $outFile";
		next;
	}

	say "making $outFile";

	my $imgSrc = Imager->new();
	$imgSrc->read(file=>$inFile)
		or die "read error. $inFile : ", $imgSrc->errstr;
	my $wSrc = $imgSrc->getwidth();
	my $hSrc = $imgSrc->getheight();
	# 出力先の画像
	my $imgDst = Imager->new(
		xsize => $wSrc * $xRepeat, 
		ysize => $hSrc * $yRepeat,
	);
	for(my $iy=0;$iy<$yRepeat; ++$iy){
		for(my $ix=0;$ix<$xRepeat; ++$ix){
			$imgDst->paste(
				src => $imgSrc,
				left => $wSrc * $ix,
				top  => $hSrc * $iy,
				width => $wSrc,
				height => $hSrc,
			);
		}
	}
	$imgDst->write( 
		file => $outFile,
		jpegquality => 90,
	) or die "$outFile : ",$imgDst->errstr;
}

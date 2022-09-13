#!/usr/bin/perl --
use strict;
use Imager;
use File::Find;
use Image::Size;
use List::Util qw(sum max);
use utf8;
use feature qw(say);

my $font = Imager::Font->new(
    file  => './HackGenNerd-Bold.ttf',
    color => '#000000',
    size  => 64,
    utf8 => 1,
    aa => 1,
);

sub caption($){
	my($text)=@_;
	my $b = $font->bounding_box(string=>$text);
	[$b,$text];
}

my %npas;
my %files;
find(
	{
		no_chdir => 1,
		wanted => sub{
			next if not -f $_;
			next if not /_npa([\d.]+).png$/;
			my($npa)=($1);
			$npas{$npa}=1;
			my($w, $h) = imgsize($_);
			$files{ "$npa" } = {
				path => $_,
				w => $w,
				h=> $h,
				npa => $npa,
			};
		},
	}
	,"outputs"
);

my @npas = sort{ (0+$a) <=> (0+$b)} keys %npas;
# my @scales = sort{ (0+$a) <=> (0+$b)} keys %scales;
say "npas=",(0+@npas);
@npas or die "no input images.\n";

my @topCaptions = map{ caption("npa $_") } @npas;
# my @leftCaptions = map{ caption("steps $_") } @steps;
# my $leftHeaderWidth = max(map{ $_->[0]->total_width }@leftCaptions);
my $topHeaderHeight = max(map{ $_->[0]->font_height }@topCaptions);
say "topHeaderHeight=$topHeaderHeight";

# 横 npas 縦 1行のみ
# $grid[y][x]
my @grid; 
my $row = [];
push @grid,$row;
for my $npa (@npas){
	push @$row, $files{ "$npa"};
}

my $gridRows = 0+@grid;
my $gridCols = 0+@{$grid[0]};

my @cellWidths;
my @cellHeights;
for(my $iy=0;$iy<$gridRows;++$iy){
	for(my $ix=0;$ix<$gridCols;++$ix){
		my $file = $grid[$iy][$ix] or next;

		my $old = $cellWidths[$ix];
		if(not $old or $old < $file->{w}){
			$cellWidths[$ix] = $file->{w};
		}

		$old = $cellHeights[$iy];
		if(not $old or $old < $file->{h}){
			$cellHeights[$iy] = $file->{h};
		}
	}
}

say "cellWidths=",join(',',@cellWidths);
say "cellHeights=",join(',',@cellHeights);

my $spacing = 8;
my $gridLeft = $spacing;
my $gridTop = $spacing + $topHeaderHeight + $spacing;
say "gridLeft=$gridLeft, gridTop=$gridTop";

# セルごとにx,y,w,hを計算する
my @cellPos;
for(my $iy=0;$iy<$gridRows;++$iy){
	for(my $ix=0;$ix<$gridCols;++$ix){
		my $pos = $cellPos[$iy][$ix] = {
			x => $ix ? $cellPos[$iy][$ix-1]->{xNext} : $gridLeft,
			y => $iy ? $cellPos[$iy-1][$ix]->{yNext} : $gridTop ,
			w => $cellWidths[$ix],
			h => $cellHeights[$iy],
		};
		$pos->{xNext} = $pos->{x} + $pos->{w} + $spacing;
		$pos->{yNext} = $pos->{y} + $pos->{h} + $spacing;
	}
}

my $totalWidth = $cellPos[0][-1]{xNext};
my $totalHeight = $cellPos[-1][0]{yNext};
say "totalWidth=$totalWidth,totalHeight=$totalHeight";

# 出力先の画像
my $imageDst = Imager->new(
	xsize => $totalWidth, 
	ysize => $totalHeight,
);

# 背景を塗りつぶす
$imageDst->box(
	color => '#ffffff', 
	filled => 1,
);

sub drawCaption{
	my($caption,$boxLeft,$boxRight,$boxTop,$boxBottom)=@_;
	$imageDst->align_string(
	    font => $font,
	    text => $caption->[1],
	    x => ($boxLeft + $boxRight)/2,
	    y => ($boxTop + $boxBottom)/2,
	    valign  =>"center",
	    halign => "center",
	);
}

# draw topCaptions
for(my $ix = 0; $ix < @topCaptions; ++$ix){
	my $pos = $cellPos[0][$ix];
	drawCaption(
		$topCaptions[$ix],
		$pos->{x},
		$pos->{x}+$pos->{w},
		$spacing,
		$gridTop -$spacing,
	);
}

# draw leftCaptions
#for(my $iy = 0; $iy < @leftCaptions; ++$iy){
#	my $pos =  $cellPos[$iy][0];
#	drawCaption(
#		$leftCaptions[$iy],
#		$spacing,
#		$gridLeft -$spacing,
#		$pos->{y},
#		$pos->{y}+$pos->{h},
#	);
#}

# グリッド中に画像を貼る
for(my $iy=0;$iy<$gridRows;++$iy){
	for(my $ix=0;$ix<$gridCols;++$ix){
		my $file = $grid[$iy][$ix] or next;
		my $pos =  $cellPos[$iy][$ix];

		my $img = Imager->new;
		$img->read(file=>$file->{path})
			or die "Cannot read: ", $img->errstr;

		$imageDst->paste(
			src => $img,
			left => $pos->{x}, 
			top  => $pos->{y},
			width => $pos->{width},
			height => $pos->{height},
		);
	}
}

# 画像を保存する
$imageDst->write(
	file => "matrix.jpg",
);

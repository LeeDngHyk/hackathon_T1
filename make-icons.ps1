Add-Type -AssemblyName System.Drawing

function Draw-Bean($g, $cx, $cy, $w, $h) {
    $body = [System.Drawing.ColorTranslator]::FromHtml('#A7CC63')
    $bodyL = [System.Drawing.ColorTranslator]::FromHtml('#B7D775')
    $sprout = [System.Drawing.ColorTranslator]::FromHtml('#7BAE3F')
    $cream = [System.Drawing.ColorTranslator]::FromHtml('#F4F6E9')
    $dark = [System.Drawing.ColorTranslator]::FromHtml('#43502C')
    $pink = [System.Drawing.ColorTranslator]::FromHtml('#F3B7A9')

    # sprout leaf on top
    $bSprout = New-Object System.Drawing.SolidBrush($sprout)
    $lw = $w*0.22; $lh = $h*0.26
    $g.FillEllipse($bSprout, [single]($cx - $lw*0.1), [single]($cy - $h*0.5 - $lh*0.55), [single]$lw, [single]$lh)

    # body
    $bBody = New-Object System.Drawing.SolidBrush($body)
    $g.FillEllipse($bBody, [single]($cx - $w/2), [single]($cy - $h/2), [single]$w, [single]$h)
    # left highlight
    $bBodyL = New-Object System.Drawing.SolidBrush($bodyL)
    $g.FillEllipse($bBodyL, [single]($cx - $w/2), [single]($cy - $h/2), [single]($w*0.5), [single]$h)

    # face
    $fw = $w*0.66; $fh = $h*0.62
    $fx = $cx - $fw/2; $fy = $cy - $fh/2 + $h*0.06
    $bCream = New-Object System.Drawing.SolidBrush($cream)
    $g.FillEllipse($bCream, [single]$fx, [single]$fy, [single]$fw, [single]$fh)

    # cheeks
    $bPink = New-Object System.Drawing.SolidBrush($pink)
    $cr = $w*0.09
    $g.FillEllipse($bPink, [single]($cx - $fw*0.42), [single]($fy + $fh*0.55), [single]$cr, [single]$cr)
    $g.FillEllipse($bPink, [single]($cx + $fw*0.42 - $cr), [single]($fy + $fh*0.55), [single]$cr, [single]$cr)

    # eyes
    $bDark = New-Object System.Drawing.SolidBrush($dark)
    $er = $w*0.06
    $ey = $fy + $fh*0.40
    $g.FillEllipse($bDark, [single]($cx - $fw*0.24 - $er/2), [single]$ey, [single]$er, [single]$er)
    $g.FillEllipse($bDark, [single]($cx + $fw*0.24 - $er/2), [single]$ey, [single]$er, [single]$er)

    # smile
    $pen = New-Object System.Drawing.Pen($dark, [single]($w*0.035))
    $pen.StartCap = 'Round'; $pen.EndCap = 'Round'
    $sw = $fw*0.34; $sh = $fh*0.22
    $g.DrawArc($pen, [single]($cx - $sw/2), [single]($fy + $fh*0.52), [single]$sw, [single]$sh, 20, 140)
}

function New-RoundedRectPath($x,$y,$w,$h,$r){
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $r*2
    $p.AddArc($x, $y, $d, $d, 180, 90)
    $p.AddArc($x+$w-$d, $y, $d, $d, 270, 90)
    $p.AddArc($x+$w-$d, $y+$h-$d, $d, $d, 0, 90)
    $p.AddArc($x, $y+$h-$d, $d, $d, 90, 90)
    $p.CloseFigure()
    return $p
}

function Make-Icon($px, $mode, $path) {
    $bmp = New-Object System.Drawing.Bitmap($px, $px)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $g.Clear([System.Drawing.Color]::Transparent)
    $green = [System.Drawing.ColorTranslator]::FromHtml('#8DBE4E')
    $bGreen = New-Object System.Drawing.SolidBrush($green)
    if ($mode -eq 'full') {
        $r = [int]($px*0.21)
        $pathObj = New-RoundedRectPath 0 0 $px $px $r
        $g.FillPath($bGreen, $pathObj)
        Draw-Bean $g ([single]($px*0.5)) ([single]($px*0.54)) ([single]($px*0.52)) ([single]($px*0.64))
    } elseif ($mode -eq 'round') {
        $g.FillEllipse($bGreen, 0, 0, [single]$px, [single]$px)
        Draw-Bean $g ([single]($px*0.5)) ([single]($px*0.54)) ([single]($px*0.52)) ([single]($px*0.64))
    } else { # fg (adaptive foreground, transparent, content in safe zone)
        Draw-Bean $g ([single]($px*0.5)) ([single]($px*0.5)) ([single]($px*0.42)) ([single]($px*0.52))
    }
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
}

$dens = @{ mdpi=1.0; hdpi=1.5; xhdpi=2.0; xxhdpi=3.0; xxxhdpi=4.0 }
$apps = @('parent','child')
foreach ($appn in $apps) {
    $resBase = "S:\Workspace\tech4good\$appn\android\app\src\main\res"
    foreach ($d in $dens.Keys) {
        $s = $dens[$d]
        $dir = "$resBase\mipmap-$d"
        Make-Icon ([int](48*$s)) 'full' "$dir\ic_launcher.png"
        Make-Icon ([int](48*$s)) 'round' "$dir\ic_launcher_round.png"
        Make-Icon ([int](108*$s)) 'fg' "$dir\ic_launcher_foreground.png"
    }
    # set adaptive background to green
    $bgxml = "$resBase\values\ic_launcher_background.xml"
    Set-Content -Path $bgxml -Encoding UTF8 -Value @'
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="ic_launcher_background">#8DBE4E</color>
</resources>
'@
    Write-Output "icons done: $appn"
}
